import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyomo
import pyomo.environ as pyo
from pyomo.util.infeasible import log_infeasible_constraints
from datetime import datetime,timedelta
import pickle
import logging
import argparse
import scipy
pd.set_option('display.max_columns', None)  # or specify a number if you want a limit
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
import os


# Set up argument parser
parser = argparse.ArgumentParser(description='Match uncontrolled shape for multi days')
parser.add_argument('--scenario_year', type=int, required=True, help='Scenario year')
parser.add_argument('--month', type=int, required=True, help='Month')
parser.add_argument('--day', type=int, required=True, help='Day')
parser.add_argument('--weekday', type=str, required=True, help='Weekday')
parser.add_argument('--monitor_hr', type=int, required=True, help='Monitor hours')

args = parser.parse_args()

# Extract arguments
scenario_year = args.scenario_year
month = args.month
day = args.day
weekday = args.weekday
monitor_hr = args.monitor_hr



grid = "369_0"
day_start_ts = pd.to_datetime(f"{scenario_year}-{month:02d}-{day:02d} 00:00:00")
nexus_day2 = day_start_ts+pd.Timedelta(hours=192)
nexus_day3 = day_start_ts+pd.Timedelta(hours=48)
nexus_day4 = nexus_day2+pd.Timedelta(hours=48)
nexus_day5 = nexus_day3+pd.Timedelta(hours=48)
nexus_day6 = nexus_day4+pd.Timedelta(hours=48)
nexus_day7 = nexus_day5+pd.Timedelta(hours=48)
day_end_ts = day_start_ts+pd.Timedelta(hours=monitor_hr)
path = f"/cluster/home/huiluo/mt/grid_{grid}/{scenario_year}_{weekday}_{day_start_ts.month:02d}_{day_start_ts.day:02d}_{monitor_hr}_uncontrolled"
data_path = f"/cluster/home/huiluo/mt/nexus_profile"
sanitycheck_path = f"{path}/sanitycheck"
num_groups = 300

TP_dict = {1:2,2:2,3:2,4:3,5:3,6:4,7:4,8:4,9:4,10:3,11:2,12:1}
TP = TP_dict[day_start_ts.month] # 1:Dec, 2:Jan,Feb,Mar,Nov, 3:Apr,May,Oct, 4:Jun,Jul,Aug,Sep
os.makedirs(path,exist_ok=True)
os.makedirs(sanitycheck_path,exist_ok=True)

###########################
#UDF
###########################
# Data Preprocessing
def calculate_next_trip_e(row):
    if next_two and (row['next_SoE_bc']+(row['next_parking_time']//60*7) < row[f'next_2_travel_TP{TP}_consumption']): # if charge with 7 KW during whole next pakring event could not cover energy requirement for the second next trip charge the part that could not be covered within this parking event
        next_trip_e = row[f'next_travel_TP{TP}_consumption']+ row[f'next_2_travel_TP{TP}_consumption'] - row['next_SoE_bc']+(row['next_parking_time']//60*7)#row['next_2_travel_TP1_consumption']
    else:
        next_trip_e =  row[f'next_travel_TP{TP}_consumption']
    return next_trip_e

def get_hour_index(ts):
    idx = int(divmod((ts-day_start_ts).total_seconds(),3600)[0]) 
    return idx

def get_soe_init(SoE_bc,arr_idx):
    if arr_idx<=0:
        arr_idx = 1
    soe_init_list = [0]*monitor_hr
    for idx in range(arr_idx):
        soe_init_list[idx] = SoE_bc
    return soe_init_list


def create_dict(start_ts,end_ts,start_hour_idx,end_hour_idx):
    # if end_hour_idx>monitor_hr-1:
    #     end_hour_idx = monitor_hr-1
    dict = [0]*monitor_hr
    for hour in range(monitor_hr):
        if start_hour_idx<hour and end_hour_idx>hour:
            dict[hour] = 60
        elif start_hour_idx==hour and end_hour_idx==hour:
            dict[hour] = end_ts.minute-start_ts.minute
        elif start_hour_idx<hour and end_hour_idx==hour:
            dict[hour] = end_ts.minute
        elif start_hour_idx==hour and end_hour_idx>hour:
            dict[hour] = 60-start_ts.minute
    return dict

def create_charge_time_list(charge_decision, start_ts, end_ts, start_hour_idx,end_hour_idx):
    # if end_hour_idx>monitor_hr-1:
    #     end_hour_idx = monitor_hr-1
    dict = [0]*monitor_hr
    if charge_decision:
        for hour in range(monitor_hr):
            if start_hour_idx<hour and end_hour_idx>hour:
                dict[hour] = 60
            elif start_hour_idx==hour and end_hour_idx==hour:
                dict[hour] = end_ts.minute-start_ts.minute
            elif start_hour_idx<hour and end_hour_idx==hour:
                dict[hour] = end_ts.minute
            elif start_hour_idx==hour and end_hour_idx>hour:
                dict[hour] = 60-start_ts.minute
    return dict


# Optimization
def create_sparse_dict(value_list,monitior_hr):
    value_indices = value_list.index.to_numpy()
    values = np.array(value_list.tolist())
    row_indices = np.repeat(value_indices,monitor_hr)
    col_indices = np.tile(np.arange(monitior_hr),len(value_indices))
    data_values = values.flatten()

    # Filter out zero values to keep the matrix sparse
    non_zero_mask = data_values != 0
    row_indices = row_indices[non_zero_mask].astype(int)
    col_indices = col_indices[non_zero_mask].astype(int)
    data_values = data_values[non_zero_mask]
    
    sparse_matrix = scipy.sparse.coo_matrix((data_values,(row_indices,col_indices)), shape=(max(value_indices)+1,monitor_hr))
    return sparse_matrix

    
# Data Postprocessing
def get_timestamp_pair(row):
    process = {}
    process_key = ()
    power = []
    for hour in range(monitor_hr):
        p_t = row['optimized_power_list'][hour]
        min_t = row['hourly_time_dict'][hour]
        if hour>0:
            min_pre, p_pre = row['hourly_time_dict'][hour-1],row['optimized_power_list'][hour-1]
        else: 
            min_pre,p_pre = 0,0
        if hour<(monitor_hr-1):
            min_next, p_next = row['hourly_time_dict'][hour+1],row['optimized_power_list'][hour+1]
        else:
            min_next, p_next = 0,0
        if p_pre==0 and p_t!=0:
            if (min_t<=60) and (min_pre>0):
                start_min = 0
            elif (min_pre==0):
                start_min = 60-min_t
            # elif (min_t == 60 or min_pre==60) and (hour!=row['arr_time_idx']):
            #     start_min = 0
            # elif (min_t<60 and min_t>0):# and (hour!=row['arr_time_idx']): #and min_pre!=60 
            #     start_min = 60-min_t
            # elif (hour==row['arr_time_idx']):
            #     start_min = row['arr_time'].minute

            else:
                start_min = 0

            start_delta_hr = day_start_ts + timedelta(hours=hour)
            start_ts = pd.Timestamp(datetime(year=start_delta_hr.year,month=start_delta_hr.month,day=start_delta_hr.day,hour=start_delta_hr.hour,minute=start_min))
            process_key = (start_ts,)
        if p_t!=0:
            power.append(p_t)
        if p_next==0 and p_t!=0:
            if (min_t==60):
                end_min=59
            elif p_pre!=0 and min_t<60:
                end_min = min_t
            elif p_pre==0 and min_t<60:
                end_min = start_min+min_t-1
            # if (min_t==60 or min_next>0) and (hour!=row['park_end_time_idx']):
            #     end_min=59
            # elif (min_t<60 and min_next==0 and min_pre>0) and (hour!=row['park_end_time_idx']):
            #     end_min=min_t
            # elif (hour==row['park_end_time_idx']):
            #     end_min=row['park_end_time'].minute
            # else:
            #     end_min = 0

            end_delta_hr = day_start_ts + timedelta(hours=hour)
            end_ts = pd.Timestamp(datetime(year=end_delta_hr.year,month=end_delta_hr.month,day=end_delta_hr.day,hour=end_delta_hr.hour,minute=end_min))

            process_key = process_key + (end_ts,)
            process[process_key] = power
            process_key = ()
            power = []
    return process

def check_outside(row):
    if (row['process_cnt']==0) and (row['st_chg_time']==row['ed_chg_time']):
        return False
    elif row['process_cnt']>=1 and (((next(iter(row['process_list'].keys())))[0]>=row['st_chg_time']) and ((next(iter(row['process_list'].keys())))[1]<=row['ed_chg_time'])):
        return False
    elif row['process_cnt']>=1 and (((next(iter(row['process_list'].keys())))[0]>=row['ed_chg_time']) or ((next(iter(row['process_list'].keys())))[1]<=row['st_chg_time'])):
        return True
    else:
        return True

def correct_st_chg_time(row):
    if row['process_list'] and row['optimized_process_mean_power']>0:
        if row['c']==True and row['st_chg_time']<day_start_ts and row['ed_chg_time']>day_start_ts and next(iter(row['process_list'].keys()))[0]==day_start_ts:
            return row['st_chg_time']
        elif next(iter(row['process_list'].keys()))[0]>=day_start_ts:
            return next(iter(row['process_list'].keys()))[0]
    elif row['process_list'] and row['optimized_process_mean_power']<0:
        return next(iter(row['process_list'].keys()))[0]
    elif (row['c']==True and row['st_chg_time']<day_start_ts and row['ed_chg_time']<day_start_ts) or (row['c']==True and row['st_chg_time']>=day_end_ts and row['ed_chg_time']>=day_end_ts):
        return row['st_chg_time']
    else:
        return None
    
def correct_ed_chg_time(row):
    if row['process_list'] and row['optimized_process_mean_power']>0:
        if row['c']==True and row['ed_chg_time']>=day_end_ts-timedelta(minutes=1) and row['st_chg_time']<day_end_ts-timedelta(minutes=1) and next(iter(row['process_list'].keys()))[1]>=day_end_ts-timedelta(minutes=1) and row['day_end_soe']<row['B']:
            return row['ed_chg_time']
        elif next(iter(row['process_list'].keys()))[1]<=day_end_ts-timedelta(minutes=1):
            return next(iter(row['process_list'].keys()))[1]
    elif row['process_list'] and row['optimized_process_mean_power']<0:
        return next(iter(row['process_list'].keys()))[1]
    elif (row['c']==True and row['st_chg_time']<day_start_ts and row['ed_chg_time']<day_start_ts) or (row['c']==True and row['st_chg_time']>=day_end_ts and row['ed_chg_time']>=day_end_ts):
        return row['ed_chg_time']
    else:
        return None
    


# Normalize Tobia's Nexus Output
hv_bus = str(89)
charge = pd.read_csv(f"{data_path}/HV_uncontrolled_load_hourly_{scenario_year}.csv",index_col=0) # in MW
charge.index = pd.to_datetime(charge.index)
day1 = pd.DataFrame(charge.loc[day_start_ts:day_start_ts+pd.Timedelta(hours=23)][hv_bus])
day2 = pd.DataFrame(charge.loc[nexus_day2:nexus_day2+pd.Timedelta(hours=23)][hv_bus])
day3 = pd.DataFrame(charge.loc[nexus_day3:nexus_day3+pd.Timedelta(hours=23)][hv_bus])
day4 = pd.DataFrame(charge.loc[nexus_day4:nexus_day4+pd.Timedelta(hours=23)][hv_bus])
day5 = pd.DataFrame(charge.loc[nexus_day5:nexus_day5+pd.Timedelta(hours=23)][hv_bus])
day6 = pd.DataFrame(charge.loc[nexus_day6:nexus_day6+pd.Timedelta(hours=23)][hv_bus])
day7 = pd.DataFrame(charge.loc[nexus_day7:nexus_day7+pd.Timedelta(hours=23)][hv_bus])
charge = pd.concat([day1, day2, day3, day4, day5, day6, day7])
period_max = charge[hv_bus].max()
charge['normalized_profile'] = charge[hv_bus]/period_max
charge.index = range(monitor_hr)




def shape_matching(i,path):
    m = pyo.ConcreteModel()
    ## Functions
    def initialize_max_p_rule(m,park,person,arr_idx,end_idx):
        return MAX_P_dict[park]
    def initialize_soe_change_rule(m,park,person,arr_idx,end_idx):
        return SOE_CHANGE_dict[park]
    def p_bound_rule(m,park,person,arr_idx,end_idx,t):
        if (park,person,arr_idx,end_idx,t) not in PARKHR_dict.keys() or (PARKHR_dict[(park,person,arr_idx,end_idx,t)]<=0):
            return (0,0)
        else:
            return (0,MAX_P_dict[park])
    def initialize_next_e_rule(m,park,person,arr_idx,end_idx):
        return NEXT_E_dict[park]
    def soe_bound_rule(m,person,t):
        return (0,EVCAP_dict[person])
    def soe_init_rule(m,person,t):
        if t==0:
            return SOE_init_dict[person]
        else:
            return 0

    ## Const
    M = 1e8
    unshifted_daily_net_charge = normalized_tot_e

    ## Set
    m.EVBAT = pyo.Set(initialize=cluster.person.unique())
    m.PARK = pyo.Set(dimen=4,initialize = park_tuple) # (Parking Event,EVBAT,Bus) Set
    m.T = pyo.Set(initialize=list(range(monitor_hr)))

    ## Params
    m.CHARGE_TARGET = pyo.Param(m.T,initialize=charge_to_match_dict)
    m.SOE_init = pyo.Param(m.EVBAT,initialize=SOE_init_dict)
    m.MAX_P = pyo.Param(m.PARK,initialize=initialize_max_p_rule)
    m.SOE_CHANGE = pyo.Param(m.PARK,initialize=initialize_soe_change_rule)
    m.PARKHR = pyo.Param(m.PARK,m.T,initialize=PARKHR_dict,default=0)
    m.PPLANED = pyo.Param(m.PARK,m.T,initialize=P_PLANED_dict,default=0)
    m.EVCAP = pyo.Param(m.EVBAT,initialize=EVCAP_dict)
    m.NEXT_E = pyo.Param(m.PARK,initialize=initialize_next_e_rule)

    ## Vars
    m.SOE = pyo.Var(m.EVBAT,m.T,within=pyo.Reals,initialize=soe_init_rule)#bounds=soe_bound_rule
    m.CHARGE_P = pyo.Var(m.PARK, m.T,within=pyo.NonNegativeReals,bounds=p_bound_rule,initialize=0) 
    m.IS_CHARGING = pyo.Var(m.PARK,m.T,within=pyo.Binary,initialize=0)
    m.IS_ACTIVE = pyo.Var(m.PARK,within=pyo.Binary,initialize=0)
    m.CHARGE_JUMP = pyo.Var(m.PARK,m.T, within=pyo.Binary,initialize=0) # detect charge jump
    m.FLEX = pyo.Var(m.PARK,m.T, within=pyo.Binary,initialize=0) # detect flexibility for each person


    ## Slack/Auxiliary
    m.DIFFER = pyo.Var(m.T,within=pyo.NonNegativeReals,bounds=(0,None),initialize=0)
    m.POWER_SHIFT_ABS = pyo.Var(m.PARK,m.T,within=pyo.NonNegativeReals,bounds=(0,None),initialize=0)
    m.BUFFER = pyo.Var(m.T,within=pyo.NonNegativeReals,initialize=0)
    m.ENERGY_DEFICIT = pyo.Var(within=pyo.NonNegativeReals,bounds=(0,0.3*unshifted_daily_net_charge),initialize=0) #Room for daily net charge energy to deviate bounds=(0,0.3*unshifted_daily_net_charge) 
    m.ENERGY_SURPLUS = pyo.Var(within=pyo.NonNegativeReals,bounds=(0,0.3*unshifted_daily_net_charge),initialize=0) #Room for daily net charge energy to deviate ,bounds=(0,0.3*unshifted_daily_net_charge)
    m.SOE_lower_slack = pyo.Var(m.EVBAT,m.T,within=pyo.NonNegativeReals,initialize=0)
    m.SOE_upper_slack = pyo.Var(m.EVBAT,m.T,within=pyo.NonNegativeReals,initialize=0)


    ############################
    def identify_charging_rule(m,park,person,arr_idx,end_idx,t):
        """
        enforce is_charging=1 if charging, 0 if discharging/no action
        """
        return m.CHARGE_P[park,person,arr_idx,end_idx,t] <= M * m.IS_CHARGING[park,person,arr_idx,end_idx,t]
    m.IdentifyCharging = pyo.Constraint(m.PARK, m.T, rule=identify_charging_rule)

    def decreasing_charge_rule(m,park,person,arr_idx,end_idx,t):
        """
        Charging power monotonically decreasing
        """
        if t>=arr_idx and t<(monitor_hr-1):
            return (m.CHARGE_P[park,person,arr_idx,end_idx,t]-m.CHARGE_P[park,person,arr_idx,end_idx,t+1])*m.IS_CHARGING[park,person,arr_idx,end_idx,t]>=0
        else:
            return pyo.Constraint.Skip
    m.DecreaseCharging = pyo.Constraint(m.PARK,m.T,rule=decreasing_charge_rule)

    def soe_update_rule(m,person,t):
        """
        Update SOE 
        """
        if t==0:
            return m.SOE[person,0]==m.SOE_init[person]
        else:
            return m.SOE[person,t]==(m.SOE[person,t-1]
                                    +sum(m.CHARGE_P[park,person,arr_idx,end_idx,t-1]*m.PARKHR[park,person,arr_idx,end_idx,t-1] for park,p,arr_idx,end_idx in m.PARK if p==person)
                                    -sum(m.SOE_CHANGE[park,p,arr_idx,end_idx] for park,p,arr_idx,end_idx in m.PARK if (p==person and arr_idx==t))
                                    )
    m.SoeUpdate = pyo.Constraint(m.EVBAT,m.T,rule=soe_update_rule)

    def soe_upper_rule(m,person,t):
        return m.SOE[person,t]<=0.95*m.EVCAP[person]+m.SOE_upper_slack[person,t]
    m.SoeUpper = pyo.Constraint(m.EVBAT,m.T,rule=soe_upper_rule)

    def soe_lower_rule(m,person,t):
        return m.SOE[person,t] + m.SOE_lower_slack[person,t]>=0.05*m.EVCAP[person]
    m.SoeLower = pyo.Constraint(m.EVBAT,m.T,rule=soe_lower_rule)

    def soe_end_rule(m,person):
        return (m.SOE[person,monitor_hr-1]
                +sum(m.CHARGE_P[park,person,arr_idx,end_idx,monitor_hr-1]*m.PARKHR[park,person,arr_idx,end_idx,monitor_hr-1] for park,p,arr_idx,end_idx in m.PARK if p==person)
                )<=m.EVCAP[person]
    m.SoeEnd = pyo.Constraint(m.EVBAT,rule=soe_end_rule)

    def next_trip_rule(m,park,person,arr_idx,end_idx,t):
        """
        SOE requirement for next trip
        """
        if end_idx==t and t<monitor_hr-1 and (end_idx!=arr_idx):
            return m.SOE[person,t]+sum(m.CHARGE_P[park,person,arr_idx,end_idx,t]*m.PARKHR[park,person,arr_idx,end_idx,t] for park,p,arr_idx,end_idx in m.PARK if p==person)>=m.NEXT_E[park,person,arr_idx,end_idx]
        else:
            return pyo.Constraint.Skip  
    m.NextTrip = pyo.Constraint(m.PARK,m.T,rule=next_trip_rule)

    def detect_charge_jump(m,park,person,arr_idx,end_idx,t):
        """
        Detect start of charging process
        """
        if t==0:
            return m.CHARGE_JUMP[park,person,arr_idx,end_idx,t]==1*m.IS_CHARGING[park,person,arr_idx,end_idx,t]
        else:
            return m.IS_CHARGING[park,person,arr_idx,end_idx,t]-m.IS_CHARGING[park,person,arr_idx,end_idx,t-1] <= M * m.CHARGE_JUMP[park,person,arr_idx,end_idx,t]
    m.DetectChargeJump = pyo.Constraint(m.PARK,m.T,rule=detect_charge_jump)


    def charge_jump_rule(m,park,p,arr_idx,end_idx):
        """
        max. 1 start of charging is allowed if parking less than 10 hr
        """
        return sum(m.CHARGE_JUMP[park,p,arr_idx,end_idx,t] for t in m.T )<=1

    m.ChargeJump = pyo.Constraint(m.PARK,rule=charge_jump_rule)


    def power_shift_rule_1(m,park,p,arr_idx,end_idx,t):
        """
        Create the slack m.POWER_SHIFT_ABS to identify wether the charging is shfited
        """
        return m.POWER_SHIFT_ABS[park,p,arr_idx,end_idx,t]>=m.CHARGE_P[park,p,arr_idx,end_idx,t]-m.PPLANED[park,p,arr_idx,end_idx,t]
    m.PowerShiftAbs1 = pyo.Constraint(m.PARK,m.T,rule=power_shift_rule_1)
        
    def power_shift_rule_2(m,park,p,arr_idx,end_idx,t):
        """
        Create the slack m.POWER_SHIFT_ABS to identify wether the charging is shfited
        """
        return m.POWER_SHIFT_ABS[park,p,arr_idx,end_idx,t]>=-(m.CHARGE_P[park,p,arr_idx,end_idx,t]-m.PPLANED[park,p,arr_idx,end_idx,t])
    m.PowerShiftAbs2 = pyo.Constraint(m.PARK,m.T,rule=power_shift_rule_2)

    def plug_in_cnt_rule(m,t):
        """
        Count plug in number at each m.T
        """
        return sum(1 if m.PARKHR[park,person,arr_idx,end_idx,t]>0 else 0 for park,person,arr_idx,end_idx in m.PARK)
    m.PlugInCnt = pyo.Expression(m.T,rule=plug_in_cnt_rule)

    def decide_flex_rule(m,park,person,arr_idx,end_idx,t):
        """
        Determine value for m.FLEX
        """
        return m.POWER_SHIFT_ABS[park,person,arr_idx,end_idx,t] <= M*m.FLEX[park,person,arr_idx,end_idx,t]
    m.DecideFlex = pyo.Constraint(m.PARK,m.T,rule=decide_flex_rule)

    def participate_rule(m,t):
        """
        Flexibility particiaption limit
        """
        return sum(m.FLEX[park,person,arr_idx,end_idx,t] for park,person,arr_idx,end_idx in m.PARK)<=0.3*m.PlugInCnt[t]# + m.BUFFER[t]
    m.ParticipateLimit = pyo.Constraint(m.T, rule=participate_rule)

    def net_charge_daily(m):
        """
        Sum of daily net charging energy
        """
        return sum(m.CHARGE_P[park,person,arr_idx,end_idx,t]*m.PARKHR[park,person,arr_idx,end_idx,t] for park,person,arr_idx,end_idx in m.PARK for t in m.T)
    m.NetChargeDaily = pyo.Expression(rule=net_charge_daily)

    def match_daily_energy_rule(m):
        """
        Match daily net charge energy as close as possible
        """
        return m.NetChargeDaily + m.ENERGY_DEFICIT == unshifted_daily_net_charge + m.ENERGY_SURPLUS
    m.MatchDailyNetCharge = pyo.Constraint(rule=match_daily_energy_rule)

    def hourly_charge_power_sum_rule(m,t):
        return sum(m.CHARGE_P[park,person,arr_idx,end_idx,t] for park,person,arr_idx,end_idx in m.PARK)
    m.HourlyChargedPowerSum = pyo.Expression(m.T,rule=hourly_charge_power_sum_rule)

    ###########################
    # Objective 
    def objective_rule(m):
        return 1000*sum((m.CHARGE_TARGET[t]-m.HourlyChargedPowerSum[t])**2 for t in m.T)+sum(m.SOE_upper_slack[person,t]+m.SOE_lower_slack[person,t] for person in m.EVBAT for t in m.T)+sum(m.BUFFER[t] for t in m.T) + sum(m.POWER_SHIFT_ABS[park,person,arr_idx,end_idx,t] for park,person,arr_idx,end_idx in m.PARK for t in m.T)
    #+sum(m.NEXT_E_slack[park,person,arr_idx,end_idx] for park,person,arr_idx,end_idx in m.PARK) 
    m.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # Step 1: Set up logging configuration
    logging.basicConfig(filename=f'{path}/infeasible_constraints.log', 
                        level=logging.INFO, 
                        format='%(asctime)s %(levelname)s: %(message)s')

    # Step 2: Define a custom function to log infeasible constraints
    def log_infeasible_constraints_to_file(model):
        # Redirect output to a string
        from io import StringIO
        import sys
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        # Call the existing function to print infeasible constraints
        log_infeasible_constraints(model)
        sys.stdout = old_stdout
        infeasible_constraints_output = mystdout.getvalue()
        logging.info(infeasible_constraints_output)

    ###########################
    # Solve model
    solver = pyo.SolverFactory('gurobi')
    solver.options['NoRelHeurTime'] = 120
    solver.options['MIPGap']=0.2
    solver.options['presolve'] = 2
    solver.options['Cuts'] = 3
    # solver.options['Method']=1
    # solver.options['NodeMethod'] = 1
    solver.options['MIPFocus'] = 3
    results = solver.solve(m,tee=False,logfile=f"{path}/cluster_{i}.log")
    print(results.solver.status)
    log_infeasible_constraints_to_file(m)
    ###############################
    # Save Results
    ch_dict = {(park, t): m.CHARGE_P[park,person,arr_idx,end_idx, t].value*emob_max_p for park,person,arr_idx,end_idx in m.PARK for t in m.T} # Denormalize back to normal power value in kW
    soe_dict = {(person,t):pyo.value(m.SOE[person,t]*emob_max_p) for person in m.EVBAT for t in m.T}
    # flex_dict = {(person,t): m.FLEX[park,person,arr_idx,end_idx,t].value for park,person,arr_idx,end_idx in m.PARK for t in m.T}
    ch = pd.Series(ch_dict).unstack()
    ch.to_csv(f'{path}/{grid}_cluster_{i}_charge.csv')

    soe = pd.Series(soe_dict).unstack()
    soe.to_csv(f'{path}/{grid}_cluster_{i}_soe.csv')
    soc = soe.apply(lambda row:row/emob_max_p/EVCAP_dict[row.name]*100,axis=1)
    soc.to_csv(f"{path}/{grid}_cluster_{i}_soc.csv")

    # person_flag = pd.Series(flex_dict).unstack()
    # person_flag.to_csv(f"{path}/{grid}_cluster_{i}_person_participate_flag.csv")
    
    # EVP_SHIFT_ABS_data = {(park,t):pyo.value(m.POWER_SHIFT_ABS[park,person,arr_idx,end_idx,t])*emob_max_p for (park,person,arr_idx,end_idx) in m.PARK for t in m.T}
    # EVP_SHIFT_ABS_df = pd.DataFrame.from_dict(EVP_SHIFT_ABS_data, orient='index',columns=['shift'])
    # EVP_SHIFT_ABS_df.index = pd.MultiIndex.from_tuples(EVP_SHIFT_ABS_df.index,names=['park','hour'])
    # EVP_SHIFT_ABS_df = EVP_SHIFT_ABS_df.unstack(level='park')
    # EVP_SHIFT_ABS_df.columns = EVP_SHIFT_ABS_df.columns.get_level_values(1)
    # EVP_SHIFT_ABS_df.to_csv(f"{path}/{grid}_cluster_{i}_power_shift.csv")

    # EVPLAN_data = {(park,t):pyo.value(m.PPLANED[park,person,arr_idx,end_idx,t])*emob_max_p for (park,person,arr_idx,end_idx) in m.PARK for t in m.T}
    # EVPLAN_data_df = pd.DataFrame.from_dict(EVPLAN_data, orient='index',columns=['plan'])
    # EVPLAN_data_df.index = pd.MultiIndex.from_tuples(EVPLAN_data_df.index,names=['park','hour'])
    # EVPLAN_data_df = EVPLAN_data_df.unstack(level='park')
    # EVPLAN_data_df.columns = EVPLAN_data_df.columns.get_level_values(1)
    # EVPLAN_data_df.to_csv(f"{path}/{grid}_cluster_{i}_plan.csv")

    return 0

# clustered = pd.read_pickle(f"{path}/grid_{grid}_{scenario_year}_{day_start_ts.month:02d}_{day_start_ts.day:02d}_{monitor_hr}_2trip_clustered_{num_groups}.pkl")
# clustered = clustered[(clustered['grid']==grid)]

for i in range(num_groups):
    cluster = pd.read_pickle(f"{path}/cluster_{i}_emob_data.pkl")
    emob_agg_e = [sum(x) for x in zip(*cluster['charge_energy_list'])] # energy in kWh
    emob_agg_p = [sum(x) for x in zip(*cluster['charge_power_list'])] # power in kW
    emob_max_p, emob_min_p,emob_tot_p = max(emob_agg_p), min(emob_agg_p), sum(emob_agg_p)
    emob_max_e, emob_min_e, emob_tot_e= max(emob_agg_e), min(emob_agg_e),sum(emob_agg_e)
    normalized_tot_e = emob_tot_e/emob_max_p
    print(f"Start Cluster {i}")

    if emob_max_p!=0:
        """
        Prepare Model Input Data
        """
        emob_agg_p_norm =  [p/emob_max_p for p in emob_agg_p]
        cluster.loc[:,'normalized_chg_power'] = cluster['chg rate']/emob_max_p
        charge_to_match_dict = charge.normalized_profile.to_dict()
        SOE_init_dict = (cluster.groupby('person')['augmented_SoE_bc'].first()/emob_max_p).to_dict()
        PARK_to_PERSON_dict = cluster.person.to_dict()
        PARK_to_ENDIDX_dict = cluster.park_end_time_idx.to_dict() # key:parking event -> value: park_end_time_idx
        PARK_to_ARRIDX_dict = cluster.arr_time_idx.to_dict() # key:parking event -> value: arr_time_idx
        park_tuple = [(PARK,int(PARK_to_PERSON_dict[PARK]),max(PARK_to_ARRIDX_dict[PARK],0),min(int(PARK_to_ENDIDX_dict[PARK]),monitor_hr-1)) for PARK,PERSON in  PARK_to_PERSON_dict.items()]
        MAX_P_dict = (cluster['chg rate']/emob_max_p).to_dict()
        # soe_change = cluster.SoE_change.apply(lambda x:max(x/emob_max_p,0))
        soe_change = cluster.SoE_change.apply(lambda x:x/emob_max_p)
        SOE_CHANGE_dict = soe_change.to_dict()
        # NEXT_E_dict = (cluster.next_trip_e/emob_max_p).to_dict()
        NEXT_E_dict = (cluster[f'next_travel_TP{TP}_consumption']/emob_max_p).to_dict()
        PARKHR_dict = {(PARK,int(PARK_to_PERSON_dict[PARK]),max(PARK_to_ARRIDX_dict[PARK],0),min(int(PARK_to_ENDIDX_dict[PARK]),monitor_hr-1),t):cluster.loc[PARK].hourly_time_dict[t]/60 for PARK in cluster.index for t in range(monitor_hr)}
        EVCAP_dict = (cluster.groupby('person').B.first()/emob_max_p).to_dict()
        
        P_PLANED_coo = create_sparse_dict(cluster.charge_power_list_sanity,monitor_hr)
        P_PLANED_dict = {(park,int(PARK_to_PERSON_dict[park]),max(PARK_to_ARRIDX_dict[park],0),min(int(PARK_to_ENDIDX_dict[park]),monitor_hr-1),t):power/emob_max_p for (park,t,power) in zip(P_PLANED_coo.row,P_PLANED_coo.col,P_PLANED_coo.data)}
        PARKHR_coo = create_sparse_dict(cluster.hourly_time_dict,monitor_hr)
        PARKHR_dict = {(park,int(PARK_to_PERSON_dict[park]),max(PARK_to_ARRIDX_dict[park],0),min(int(PARK_to_ENDIDX_dict[park]),monitor_hr-1),t):minutes/60 for  park,t,minutes in zip(PARKHR_coo.row,PARKHR_coo.col,PARKHR_coo.data)}

        opt_res_code = shape_matching(i,path)


concat_charge = pd.DataFrame()
concat_soe = pd.DataFrame()
concat_flexperson = pd.DataFrame()
concat_soc = pd.DataFrame()
for i in range(num_groups):
    charge_i = pd.read_csv(f'{path}/{grid}_cluster_{i}_charge.csv')
    concat_charge = pd.concat([concat_charge,charge_i],axis=0)
    soe_i = pd.read_csv(f'{path}/{grid}_cluster_{i}_soe.csv')
    concat_soe = pd.concat([concat_soe,soe_i],axis=0)
    soc_i = pd.read_csv(f'{path}/{grid}_cluster_{i}_soc.csv')
    concat_soc = pd.concat([concat_soc,soc_i],axis=0)
    # flexperson_i = pd.read_csv(f'{path}/{grid}_cluster_{i}_person_participate_flag.csv')
    # concat_flexperson = pd.concat([concat_flexperson,flexperson_i],axis=0)
    
concat_charge = concat_charge.rename(columns={'Unnamed: 0':'event_index'})
concat_soe = concat_soe.rename(columns={'Unnamed: 0':'person'})
concat_soc = concat_soc.rename(columns={'Unnamed: 0':'person'})
concat_soc.set_index('person',inplace=True)
# concat_flexperson = concat_flexperson.rename(columns={'Unnamed: 0':'person'})

concat_res = concat_charge
concat_res.loc[:,'event_index'] = concat_charge['event_index']
# concat_charge.set_index('event_index',inplace=True)
concat_res.set_index('event_index', inplace=True)


# Cast to 0 is power too low
for column in concat_res.columns:
    if column != 'event_index':
        # Apply the condition and replace values
        concat_res.loc[:, column] = concat_res[column].apply(lambda x: 0 if abs(x) < 0.01 else x)
        
concat_charge.to_csv(f'{path}/concat_charge_power_all.csv')
concat_res.to_csv(f'{path}/concat_net_power_all.csv')
concat_soe.to_csv(f'{path}/concat_soe_all.csv')
concat_soc.to_csv(f'{path}/concat_soc_all.csv')
# concat_flexperson.to_csv(f'{path}/concat_flexperson_all.csv')



