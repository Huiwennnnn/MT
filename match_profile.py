import pandas as pd
import matplotlib.pyplot as plt
from pyomo.environ import *

mobility = pd.read_csv("/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/mobility/grid369_mobility_dataset.csv")
mobility['dep_time'] = pd.to_datetime(mobility['dep_time'],format='mixed')
mobility['arr_time'] = pd.to_datetime(mobility['arr_time'],format='mixed')
mobility['st_chg_time'] = pd.to_datetime(mobility['st_chg_time'],format='mixed')
mobility['ed_chg_time'] = pd.to_datetime(mobility['ed_chg_time'],format='mixed')
mobility['chg_time'] = mobility['ed_chg_time']-mobility['st_chg_time']

mobility['dep_hour'] = mobility['dep_time'].dt.hour
mobility['arr_hour'] = mobility['arr_time'].dt.hour
mobility.sort_values(by=['person','dep_time'])
mobility['next_travel_TP1_consumption'] = mobility.groupby('person')['TP1 consumption kWh'].shift(-1).fillna(0)
inbound = mobility[(mobility['grid']=="369_0") & (mobility['type_day']=="Friday")]
inbound['arr_time'] = inbound['dep_time'] + pd.to_timedelta(inbound['trav_time'], unit='m')
inbound['arr_time'] = inbound['arr_time'].apply(lambda dt:dt.replace(day=6,month=5,year=2050))
inbound['st_chg_time'] = inbound['st_chg_time'].apply(lambda dt:dt.replace(day=6,month=5,year=2050))
inbound['ed_chg_time'] = inbound['st_chg_time']+inbound['chg_time']

inbound['park_end_time'] = inbound['arr_time']+pd.to_timedelta(inbound['parking_time'],unit='m')
inbound['park_end_hour'] = inbound['park_end_time'].dt.hour
inbound['park_end_day'] = inbound['park_end_time'].dt.day
inbound.insert(0,'event_index',inbound.index)

date = 6


def create_dict(row):
    dict = {}
    for hour in range(24):
        if ((row['park_end_day'] == date) & (row['park_end_hour'] > hour) & (row['arr_hour'] < hour)) | (
                (row['park_end_day'] > date) & (row['arr_hour'] < hour)):
            dict[hour] = 60
        elif ((row['park_end_day'] == date) & (row['park_end_hour'] > hour) & (row['arr_hour'] == hour)) | (
                (row['park_end_day'] > date) & (row['arr_hour'] == hour)):
            dict[hour] = 60 - row['arr_time'].minute
        elif (row['park_end_hour'] == hour) & (row['arr_hour'] < hour) & (row['park_end_day'] == date):
            dict[hour] = row['park_end_time'].minute
        elif (row['park_end_hour'] == hour) & (row['park_end_day'] == date) & (row['arr_hour'] == hour):
            dict[hour] = row['park_end_time'].minute - row['arr_time'].minute
        else:
            dict[hour] = 0
    return dict


def soe_init(row):
    soe_init = {}
    for hour in range(24):
        if hour < row["arr_hour"]:
            soe_init[hour] = row['SoE_bc']
        else:
            soe_init[hour] = 0
    return soe_init


def create_charge_time_list(row):
    charge_time_list = [0] * 24
    if row['c']:
        st_chg = row['st_chg_time'].hour
        ed_chg = row['ed_chg_time'].hour
        for t in range(24):
            if (st_chg == t) & (ed_chg == t) & (row['ed_chg_time'].day == date):
                charge_time_list[t] = row['ed_chg_time'].minute - row['st_chg_time'].minute
            elif (st_chg < t) & (ed_chg > t) & (row['ed_chg_time'].day == date):
                charge_time_list[t] = 60
            elif (st_chg < t) & (ed_chg == t) & (row['ed_chg_time'].day == date):
                charge_time_list[t] = row['ed_chg_time'].minute
            elif (st_chg == t) & (((ed_chg > t) & (row['ed_chg_time'].day == date)) | (row['ed_chg_time'].day > date)):
                charge_time_list[t] = 60 - row['st_chg_time'].minute
            elif (st_chg < t) & (row['ed_chg_time'].day > date):
                charge_time_list[t] = 60
            else:
                charge_time_list[t] = 0
    return charge_time_list


inbound['max_chg_e'] = inbound['B'] - inbound['SoE_bc']
inbound['real_chg_e'] = inbound['SoE_ac'] - inbound['SoE_bc']
inbound['hourly_time_dict'] = inbound.apply(lambda x: create_dict(x), axis=1)
inbound['soe_init'] = inbound.apply(lambda x: soe_init(x), axis=1)
inbound['charge_time_list'] = inbound.apply(lambda x: create_charge_time_list(x), axis=1)
inbound['charge_power_list'] = inbound.apply(lambda x: [x['chg rate'] if t > 30 else 0 for t in x['charge_time_list']],
                                             axis=1)
inbound['charge_energy_list'] = inbound.apply(lambda x: [t / 60 * x['chg rate'] for t in x['charge_time_list']], axis=1)

# Normalize the mobility aggregated charge power, calculate min, max, sum of power
emob_agg_e = [sum(x) for x in zip(*inbound['charge_energy_list'])] # energy in kWh
emob_agg_p = [sum(x) for x in zip(*inbound['charge_power_list'])] # power in kW
emob_max_p, emob_min_p = max(emob_agg_p), min(emob_agg_p)
emob_tot_p = sum(emob_agg_p)

emob_max_e, emob_min_e = max(emob_agg_e), min(emob_agg_e)
emob_tot_e = sum(emob_agg_e)

print("Mobility data peak power:",emob_max_p,"Mobility data minimal power:",emob_min_p,"Mobility data total power:",emob_tot_p)
print("Mobility data peak energy:",emob_max_e,"Mobility data minimal energy:",emob_min_e,"Mobility data total energy:",emob_tot_e)

emob_agg_p_norm =  [p/emob_max_p for p in emob_agg_p]
plt.plot(emob_agg_p,label='power')
plt.plot(emob_agg_e,label='energy')
plt.legend()
inbound['normlaized_chg_power'] = inbound['chg rate']/emob_max_p
normalized_tot_e = emob_tot_e/emob_max_p
print(normalized_tot_e)


# Normalize Tobia's Nexus Output
hv_bus = str(89)
tomatch = pd.read_csv(f"/Users/huiwen/Library/Mobile Documents/com~apple~CloudDocs/Thesis/extracted_data/map_bus/HV_emob_load_2050.csv") # in MW
tomatch.rename(columns={'Unnamed: 0':'ts'}, inplace=True)
tomatch = tomatch.loc[(tomatch.ts<"2050-05-07 00:00:00") & (tomatch.ts>="2050-05-06 00:00:00")][['ts','peak',hv_bus]]
day_max = tomatch[hv_bus].max()
day_min = tomatch[hv_bus].min()

tomatch['normalized_profile']=tomatch[hv_bus]/day_max
tomatch.index=range(24)
tomatch.plot()







#######################################################################################
#            Optimize aggregated demand profile to fit nexus output                   #
#######################################################################################
# No smart charging
m = AbstractModel()
############################
# Set
m.E = Set() # parking event set
m.T = Set() # hour of the day
############################
# Decision Variable
m.charge_power = Var(m.E, m.T,within=Reals,bounds=(-1,1),initialize=0) # positive for charge, negative for discharge, 0 for parked/plugged in without charging
m.ax_charge_power = Var(m.E, m.T, within=NonNegativeReals, bounds=(0, 1), initialize=0)
m.is_parked = Var(m.E, m.T, within=Binary)
m.is_charging = Var(m.E,m.T,within=Binary)
# m.z_power_limit= Var(m.E,m.T,within=Binary,initialize=0) # Binary variable for charge power limit

###########################
# Parameters
m.bigM = Param(initialize=1e8) # bigM
m.load_to_match = Param(m.T) # Normalized nexus profile
m.origin_load_to_match = Param(m.T) # Origin Scale nexus profile
m.max_power = Param(m.E) # unit kW
m.parking_time = Param(m.E,m.T) # parking minutes within this hour
m.next_trip_e = Param(m.E) # normalized energy next trip requires
m.park_end_hour = Param(m.E) # Park End Hour
m.SoE = Param(m.E,m.T) # initial SoE profile
m.max_chg_e = Param(m.E) # max energy to charge for each event
m.unshifted_daily_energy = Param()

############################
# vehicles not parked/charging power upper limit
def not_parked_rule(m,e,t):
    if m.parking_time[e,t]==0:
        return m.charge_power[e,t]==0
    else:
        return Constraint.Skip
m.not_parked = Constraint(m.E,m.T, rule=not_parked_rule)

# Enforce binary varialbe represent parking status
def parking_logic_constraint_1(m, e, t):
    M = 1000000  # Big M value, adjust as necessary
    return m.parking_time[e, t] <= M * m.is_parked[e, t]
m.parking_logic_1 = Constraint(m.E, m.T, rule=parking_logic_constraint_1)
def parking_logic_constraint_2(m, e, t):
    M = 1000000  # Big M value, adjust as necessary
    return m.parking_time[e, t] >= M * (1-m.is_parked[e, t])
m.parking_logic_2 = Constraint(m.E, m.T, rule=parking_logic_constraint_2)
# enforce absolute value of charge_power
def ax_charge_power_constraint_upper(m, e, t):
    return m.ax_charge_power[e, t] >= m.charge_power[e, t]
def ax_charge_power_constraint_lower(m, e, t):
    return m.ax_charge_power[e, t] >= -m.charge_power[e, t]
m.ax_charge_power_upper = Constraint(m.E, m.T, rule=ax_charge_power_constraint_upper)
m.ax_charge_power_lower = Constraint(m.E, m.T, rule=ax_charge_power_constraint_lower)




# Enforce binary varialbe represent parking status
def charging_logic_constraint_1(m, e, t):
    M = 1000000  # Big M value, adjust as necessary
    return m.ax_charge_power[e, t] <= M * m.is_charging[e, t]
m.charging_logic_1 = Constraint(m.E, m.T, rule=charging_logic_constraint_1)
def charging_logic_constraint_2(m, e, t):
    M = 1000000  # Big M value, adjust as necessary
    return m.ax_charge_power[e, t] >= M * (1-m.is_charging[e, t])
m.charging_logic_2 = Constraint(m.E, m.T, rule=charging_logic_constraint_2)


# absolute power limit
def max_power_constraint(m, e, t):
    return m.ax_charge_power[e, t] <= m.max_power[e] # smaller than max power
def enforce_abs_power_min(m, e, t):
    return m.ax_charge_power[e, t] >= 0.7 * m.max_power[e] * m.is_parked[e, t] * m.is_charging[e,t] # deviate max. 50% from max power
m.max_power_limit = Constraint(m.E, m.T, rule=max_power_constraint)
m.abs_power_min_constraint = Constraint(m.E, m.T, rule=enforce_abs_power_min)

'''
SoE non-negative and prepare for future trips constraints
'''
# update SoE for event e at hour t
def SoE_update_rule(m,e,t):
    if t==0:
        return m.SoE[e,t]
    else:
        return m.SoE[e,t-1]+m.charge_power[e,t-1]*m.parking_time[e,t-1]
m.SoE_update=Expression(m.E,m.T,rule=SoE_update_rule)

# Prepare for next trip:
# def next_trip_min_SoE_rule(m,e,t):
#     if t==m.park_end_hour[e]:
#         return m.SoE_update[e,t]>=m.next_trip_e[e]
#     else:
#         return Constraint.Skip
# m.next_trip_min_SoE = Constraint(m.E,m.T,rule=next_trip_min_SoE_rule)

# SoE shouldn't be negative:
def SoE_nonnegative_rule(m,e,t):
    return m.SoE_update[e,t]>=0
m.SoE_nonnegative = Constraint(m.E,m.T, rule=SoE_nonnegative_rule)

'''
Charging Limit considering battery capacity
'''
# total energy requested from grid for this parking event
def tot_e_required_rule(m,e):
    return sum(m.charge_power[e,t]*m.parking_time[e,t] for t in m.T)
m.tot_e_required=Expression(m.E, rule=tot_e_required_rule)

# Max. energy can be charged for each parked EV
def max_e_required_rule(m,e):
    return m.tot_e_required[e]<=m.max_chg_e[e]
m.max_e_required = Constraint(m.E,rule=max_e_required_rule)

'''
Expression for hourly aggregated power
'''
# hourly charge power requested from the grid for all EVs
def hourly_tot_charge_power_rule(m,t):
    return sum(m.charge_power[e,t] for e in m.E)
m.hourly_tot_charge_power = Expression(m.T,rule=hourly_tot_charge_power_rule)

'''
Energy Matching
'''
# houlry charge energy requested from the grid for all EVs of shifted profile
def hourly_tot_charge_energy_rule(m,t):
    return sum(m.charge_power[e,t]*m.parking_time[e,t] for e in m.E)
m.hourly_tot_charge_energy = Expression(m.T, rule=hourly_tot_charge_energy_rule)

# Net charged energy for whole day of shifted profile
def net_charge_daily_rule(m):
    return sum(m.hourly_tot_charge_energy[t] for t in m.T)
m.shifted_daily_energy = Expression(rule=net_charge_daily_rule)

# Match shifted daily required energy with the unshifted sum
def match_daily_energy_rule(m):
    return m.shifted_daily_energy==m.unshifted_daily_energy
m.match_daily_energy = Constraint(rule=match_daily_energy_rule)


###########################
# Objective
def objective_rule(m):
    return sum((m.load_to_match[t]-m.hourly_tot_charge_power[t])**2 for t in m.T)
m.objective = Objective(rule=objective_rule, sense=minimize)
###########################
# Solve abstract model
tomatch_t = tomatch['normalized_profile']
tomatch_origin = tomatch[hv_bus]
t_list = list(range(24))
e_list = list(inbound.event_index)
data_t = {None:{
        'E': {None:e_list},
        'T': {None:t_list},
        'max_power':{e:inbound.loc[e,'normlaized_chg_power'] for e in e_list},
        'parking_time': {(e,t):inbound.loc[e,'hourly_time_dict'][t]/60 for e in e_list for t in t_list},
        'next_trip_e': {e:inbound.loc[e,'next_travel_TP1_consumption']/emob_max_p for e in e_list},
        'park_end_hour':{e:inbound.loc[e,'park_end_hour'] for e in e_list},
        'SoE':{(e,t):inbound.loc[e,'soe_init'][t]/emob_max_p for e in e_list for t in t_list},
        'max_chg_e':{e:inbound.loc[e,'max_chg_e']/emob_max_p for e in e_list},
        'load_to_match':{t:tomatch_t[t] for t in t_list},
        'unshifted_daily_energy':{None:normalized_tot_e},
}}
instance_t = m.create_instance(data=data_t)
# instance_t.pprint()
solver = SolverFactory('gurobi')
# solver.options['tol']= 1e-4
# solver.options['max_iter'] = 8000
solver.solve(instance_t,tee=True,keepfiles=True,logfile="match_profile_log.log")
# log_infeasible_constraints(m, log_expression=True, log_variables=True)
# logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.INFO)


data_for_df = {(e, t): instance_t.charge_power[e, t].value for e in instance_t.E for t in instance_t.T}
# Convert the dictionary into a multi-index series to facilitate unstacking
multi_index_series = pd.Series(data_for_df).unstack()
# Save the restructured data to CSV
multi_index_series.to_csv('test_ipopt.csv')