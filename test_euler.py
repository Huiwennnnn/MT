import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyomo.environ import *
from datetime import datetime
import pickle

# Basic Setting
this_day = '2050-05-06'
this_weekday = 'Friday'
user = 'huiluo'
grid_folder = 'grid_369_0'
path = f"/cluster/home/{user}/mt/{grid_folder}"
data_path = f"/cluster/home/{user}/mt/nexus_profile"

# Utlity
date = 6


# Data Preprocessing
def create_dict(row):
    dict = [0] * 24
    for hour in range(24):
        if (row['park_end_day'] == date) & (row['park_end_hour'] > hour) & (row['arr_time'].day == date) & (
                row['arr_hour'] < hour):
            dict[hour] = 60
        elif (row['park_end_day'] == date) & (row['park_end_hour'] > hour) & (row['arr_time'].day < date):
            dict[hour] = 60
        elif (row['park_end_day'] > date) & (row['arr_time'].day == date) & (row['arr_hour'] < hour):
            dict[hour] = 60
        elif (row['park_end_day'] > date) & (row['arr_time'].day < date):
            dict[hour] = 60
        elif (row['park_end_day'] == date) & (row['park_end_hour'] > hour) & (row['arr_time'].day == date) & (
                row['arr_hour'] == hour):
            dict[hour] = 60 - row['arr_time'].minute
        elif (row['park_end_day'] > date) & (row['arr_hour'] == hour) & (row['arr_time'].day == date):
            dict[hour] = 60 - row['arr_time'].minute
        elif (row['park_end_hour'] == hour) & (row['park_end_day'] == date) & (row['arr_hour'] < hour) & (
                row['arr_time'].day == date):
            dict[hour] = row['park_end_time'].minute
        elif (row['park_end_hour'] == hour) & (row['park_end_day'] == date) & (row['arr_time'].day < date):
            dict[hour] = row['park_end_time'].minute
        elif (row['park_end_hour'] == hour) & (row['park_end_day'] == date) & (row['arr_hour'] == hour) & (
                row['arr_time'].day == date):
            dict[hour] = row['park_end_time'].minute - row['arr_time'].minute
        else:
            dict[hour] = 0
    return dict


def soe_init(row):
    soe_init = [0] * 24
    for hour in range(24):
        if row['arr_time'].day < date:
            soe_init[hour] = row['SoE_bc']
        elif (hour <= row["arr_hour"]) and (row['arr_time'].day == date):
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


# Data Postprocessing
def get_timestamp_pair(row):
    process = {}
    process_key = ()
    power = []
    for hour in range(24):
        p_t = row['optimized_power_list'][hour]
        min_t = row['hourly_time_dict'][hour]
        if hour > 0:
            min_pre, p_pre = row['hourly_time_dict'][hour - 1], row['optimized_power_list'][hour - 1]
        else:
            min_pre, p_pre = 0, 0
        if hour < 23:
            min_next, p_next = row['hourly_time_dict'][hour + 1], row['optimized_power_list'][hour + 1]
        else:
            min_next, p_next = 0, 0
        if p_pre == 0 and p_t != 0:
            if (min_t == 60 or min_pre == 60) and (hour != row['arr_time'].hour):
                start_min = 0
            elif (min_t < 60 and min_pre != 60 and min_t > 0) and (hour != row['arr_time'].hour):
                start_min = 60 - min_t
            elif (hour == row['arr_time'].hour):
                start_min = row['arr_time'].minute
            else:
                start_min = 0
            start_ts = pd.Timestamp(datetime(year=2050, month=5, day=date, hour=hour, minute=start_min))
            process_key = (start_ts,)
        if p_t != 0:
            power.append(p_t)
        if p_next == 0 and p_t != 0:
            if (min_t == 60 or min_next > 0) and (hour != row['park_end_time'].hour):
                end_min = 59
            elif (min_t < 60 and min_next == 0 and min_pre > 0) and (hour != row['park_end_time'].hour):
                end_min = min_t
            elif (hour == row['park_end_time'].hour):
                end_min = row['park_end_time'].minute
            else:
                end_min = 0
            end_ts = pd.Timestamp(datetime(year=2050, month=5, day=date, hour=hour, minute=end_min))
            process_key = process_key + (end_ts,)
            process[process_key] = power
            process_key = ()
            power = []
    return process


df = pd.read_csv(f"{path}/grid369_mobility_dataset.csv")
df['dep_time'] = pd.to_datetime(df['dep_time'])
df['arr_time'] = pd.to_datetime(df['arr_time'])
df['st_chg_time'] = pd.to_datetime(df['st_chg_time'])
df['ed_chg_time'] = pd.to_datetime(df['ed_chg_time'])
df['chg_time'] = pd.to_timedelta(df['ed_chg_time'] - df['st_chg_time'], unit='m')

df['dep_hour'] = df['dep_time'].dt.hour
df['arr_hour'] = df['arr_time'].dt.hour
df.sort_values(by=['person', 'dep_time'])
df['next_travel_TP1_consumption'] = df.groupby('person')['TP1 consumption kWh'].shift(-1).fillna(0)

d = df[(df['grid'] == "369_0") & ((df['type_day'] == 'Thursday') | (df['type_day'] == 'Friday'))]
d['arr_time'] = d['dep_time'] + pd.to_timedelta(d['trav_time'], unit='m')
d.loc[d['type_day'] == 'Friday', 'arr_time'] = d.loc[d['type_day'] == 'Friday', 'arr_time'].apply(
    lambda dt: dt.replace(day=6, month=5, year=2050))
d.loc[d['type_day'] == 'Thursday', 'arr_time'] = d.loc[d['type_day'] == 'Thursday', 'arr_time'].apply(
    lambda dt: dt.replace(day=5, month=5, year=2050))
d['park_end_time'] = d['arr_time'] + pd.to_timedelta(d['parking_time'], unit='m')
d['park_end_hour'] = d['park_end_time'].dt.hour
d['park_end_day'] = d['park_end_time'].dt.day
d.loc[:, 'st_chg_time'] = d.apply(
    lambda row: row['st_chg_time'].replace(day=row['arr_time'].day, month=row['arr_time'].month,
                                           year=row['arr_time'].year), axis=1)
d['ed_chg_time'] = d['st_chg_time'] + d['chg_time']
d['chg_time'] = d['chg_time'].dt.total_seconds() / 60
d = d[(d['park_end_time'] >= '2050-05-06 00:00:00') & (d['arr_time'] < '2050-05-07 00:00:00')]
d.insert(0, 'event_index', d.index)
d['max_chg_e'] = d['B'] - d['SoE_bc']
d['real_chg_e'] = d['SoE_ac'] - d['SoE_bc']
d['hourly_time_dict'] = d.apply(lambda x: create_dict(x), axis=1)
d['soe_init'] = d.apply(lambda x: soe_init(x), axis=1)
d['charge_time_list'] = d.apply(lambda x: create_charge_time_list(x), axis=1)
d['charge_power_list'] = d.apply(lambda x: [x['chg rate'] if t > 30 else 0 for t in x['charge_time_list']], axis=1)
d['charge_energy_list'] = d.apply(lambda x: [t / 60 * x['chg rate'] for t in x['charge_time_list']], axis=1)


# Normalize Tobia's Nexus Output
hv_bus = str(89)
# Controlled charging
charge_raw = pd.read_csv(f"{data_path}/Added_up_charge_2050_raw.csv",index_col=0)  # in MW
print("columns' names: \n")
print(charge_raw.columns)
charge_raw['ts'] = pd.to_datetime(charge_raw['ts'])
print("charge_raw unfiltered: \n")
print(charge_raw.head())
charge = charge_raw[(charge_raw.ts < pd.to_datetime("2050-05-07 00:00:00")) & (charge_raw.ts >= pd.to_datetime("2050-05-06 00:00:00"))][['ts', 'peak', hv_bus]]
charge.index = range(24)
print("charge_raw filtered: \n")
print(charge.head())
print(len(charge))

# Controlled discharging
discharge_raw = pd.read_csv(f"{data_path}/EVBatt_power_hourly_2050_discharge_mapped_raw.csv", index_col=0)
print("columns' names: \n")
print(charge_raw.columns)
discharge_raw['ts'] = pd.to_datetime(discharge_raw['ts'])
print("discharge_raw unfiltered: \n")
print(discharge_raw.head())
discharge = discharge_raw.loc[(discharge_raw.ts < pd.to_datetime("2050-05-07 00:00:00")) & (discharge_raw.ts >= pd.to_datetime("2050-05-06 00:00:00"))][
    ['ts', hv_bus]]
discharge.index = range(24)
# Find Netload Max
net = charge['89'] - discharge['89']
print(discharge['89'])
print(pd.__version__)
print("net: \n", net)

print("net-type: ", net.dtype)
day_max = net.max()
print("day_max: \n", day_max)
charge['normalized_profile'] = charge[hv_bus] / day_max
discharge['normalized_profile'] = discharge[hv_bus] / day_max
net = charge['89'] - discharge['89']
net_normalized = net / day_max

print(charge['normalized_profile'])
print(discharge['normalized_profile'])

clustered = d.sort_values(by=['arr_time', 'parking_time'])
k = 25
group = np.arange(len(clustered)) % k
clustered['cluster'] = group


def opt_charge_profile(charge, discharge, net_normalized, cluster, emob_max_p, normalized_tot_e, date, cluster_i, path):
    t_list = list(range(24))
    e_list = list(cluster.event_index)
    tomatch_c = charge['normalized_profile']
    tomatch_d = discharge['normalized_profile']
    tomatch_net = net_normalized

    m = ConcreteModel()
    ############################
    # Set
    m.E = Set(initialize=e_list)  # parking event set
    m.T = Set(initialize=t_list)  # hour of the day
    ############################
    # Decision Variable
    m.charge_power = Var(m.E, m.T, within=NonNegativeReals, bounds=(0, 1.5), initialize=0)
    m.discharge_power = Var(m.E, m.T, within=NonNegativeReals, bounds=(0, 1), initialize=0)

    m.is_parked = Var(m.E, m.T, within=Binary)
    m.is_charging = Var(m.E, m.T, within=Binary, initialize=0)
    m.is_discharging = Var(m.E, m.T, within=Binary, initialize=0)
    m.is_active = Var(m.E, within=Binary, initialize=0)
    m.charge_jump = Var(m.E, m.T, within=Binary, initialize=0)  # detect charge jump
    m.discharge_jump = Var(m.E, m.T, within=Binary, initialize=0)  # detect discharge jump

    # Slack Variable
    m.charge_power_limit_s = Var(m.E, m.T, within=NonNegativeReals, initialize=0)
    m.discharge_power_limit_s = Var(m.E, m.T, within=NonNegativeReals, initialize=0)
    m.energy_surplus = Var(within=NonNegativeReals, bounds=(0, 0.1 * normalized_tot_e))
    m.energy_deficit = Var(within=NonNegativeReals, bounds=(0, 0.1 * normalized_tot_e))

    # Auxiliary Variable
    m.tot_charge_z = Var(m.T, within=NonNegativeReals, initialize=0)
    m.tot_discharge_z = Var(m.T, within=NonNegativeReals, initialize=0)
    ###########################
    # Parameters
    load_c = {t: tomatch_c[t] for t in t_list}
    m.charge_to_match = Param(m.T, initialize=load_c)  # Normalized nexus profile

    load_d = {t: tomatch_d[t] for t in t_list}
    m.discharge_to_match = Param(m.T, initialize=load_d)  # Normalized nexus profile

    load_net = {t: tomatch_net[t] for t in t_list}
    m.netload_to_match = Param(m.T, initialize=load_net)  # Normalized net charge nexus profile

    capacity = {e: cluster.loc[e, 'B'] / emob_max_p for e in e_list}
    m.capacity = Param(m.E, initialize=capacity)  # Normalized battery capacity

    max_power_dict = {e: cluster.loc[e, 'normalized_chg_power'] for e in e_list}  # max charge rate at normalized scale
    m.max_power = Param(m.E, initialize=max_power_dict)  # unit kW

    parking_time_dict = {(e, t): cluster.loc[e, 'hourly_time_dict'][t] / 60 for e in e_list for t in t_list}
    m.parking_time = Param(m.E, m.T, initialize=parking_time_dict)  # parking minutes within this hour

    next_trip_e_dict = {e: cluster.loc[e, 'next_travel_TP1_consumption'] / emob_max_p for e in e_list}
    m.next_trip_e = Param(m.E, initialize=next_trip_e_dict)  # normalized energy next trip requires

    park_end_hour_dict = {e: cluster.loc[e, 'park_end_hour'] for e in e_list}
    m.park_end_hour = Param(m.E, initialize=park_end_hour_dict)  # Park End Hour

    park_end_day = {e: cluster.loc[e, 'park_end_day'] for e in e_list}
    m.park_end_day = Param(m.E, initialize=park_end_day)  # Park End Day

    arr_day = {e: cluster.loc[e, 'arr_time'].day for e in e_list}
    m.arr_day = Param(m.E, initialize=arr_day)

    arr_hour = {e: cluster.loc[e, 'arr_hour'] for e in e_list}
    m.arr_hour = Param(m.E, initialize=arr_hour)

    SoE_dict = {(e, t): cluster.loc[e, 'soe_init'][t] / emob_max_p for e in e_list for t in t_list}
    m.SoE_2d = Param(m.E, m.T, initialize=SoE_dict)  # initial SoE profile

    SoE_init = {e: cluster.loc[e, 'SoE_bc'] / emob_max_p for e in e_list}
    m.SoE_init = Param(m.E, initialize=SoE_init)  # initial SoE profile

    max_chg_e_dict = {e: cluster.loc[e, 'max_chg_e'] / emob_max_p for e in e_list}
    m.max_chg_e = Param(m.E, initialize=max_chg_e_dict)  # max energy to charge for each event

    m.unshifted_daily_energy = Param(initialize=normalized_tot_e)

    charge_status_change = {e: 0 for e in e_list}
    m.charge_status_change = Param(m.E, initialize=charge_status_change)

    ############################
    # enforce is_charging=1 if charging, 0 if discharging/no action
    def positive_charge_power_rule(m, e, t):
        return m.charge_power[e, t] <= 1000000 * m.is_charging[e, t]

    m.positive_charge_power_con = Constraint(m.E, m.T, rule=positive_charge_power_rule)

    # enforce is_discharging=1 if discharging, 0 if charging/no action
    def negative_charge_power_rule(m, e, t):
        return m.discharge_power[e, t] <= 1000000 * m.is_discharging[e, t]

    m.negative_charge_power_con = Constraint(m.E, m.T, rule=negative_charge_power_rule)

    # avoid simultaneous charge and discharge
    def non_simultaneous_rule(m, e, t):
        return m.is_charging[e, t] + m.is_discharging[e, t] <= 1

    m.non_simultaneous_con = Constraint(m.E, m.T, rule=non_simultaneous_rule)

    # enforce is_parked=1 if parking and ready to participate in grid, 0 if not available
    def parking_logic_constraint(m, e, t):
        M = 1000000  # Big M value, adjust as necessary
        return m.parking_time[e, t] <= M * m.is_parked[e, t]

    m.parking_logic = Constraint(m.E, m.T, rule=parking_logic_constraint)

    '''
    charging power limit
    '''

    # charge/discharge power slightly deviate from max charge power
    def charge_power_limit_rule(m, e, t):
        return m.charge_power[e, t] + m.charge_power_limit_s[e, t] == m.max_power[e] * m.is_charging[e, t]

    m.power_limit_c = Constraint(m.E, m.T, rule=charge_power_limit_rule)

    def discharge_power_limit_rule(m, e, t):
        return m.discharge_power[e, t] + m.discharge_power_limit_s[e, t] == m.max_power[e] * m.is_discharging[e, t]

    m.power_limit_d = Constraint(m.E, m.T, rule=discharge_power_limit_rule)

    def parking_rule_c(m, e, t):
        if m.parking_time[e, t] == 0:
            return m.charge_power[e, t] == 0  # not parked for charge
        elif t < 23 and m.parking_time[e, t] < 0.5 and m.parking_time[e, t + 1] == 0:
            return m.charge_power[e, t] == 0  # do not charge in the last parking hour less than 30 min
        else:
            return Constraint.Skip

    m.parking_c = Constraint(m.E, m.T, rule=parking_rule_c)

    def parking_rule_d(m, e, t):
        if m.parking_time[e, t] == 0:
            return m.discharge_power[e, t] == 0  # not parked for discharge
        else:
            return Constraint.Skip

    m.parking_d = Constraint(m.E, m.T, rule=parking_rule_d)

    # Plug in as soon as parked
    def charge_immediate_rule(m, e, t):
        if (t > m.arr_hour[e]) and (m.arr_day[e] == date) and t > 0:
            return m.is_charging[e, t - 1] >= m.is_charging[e, t]
        elif m.arr_day[e] < date and t > 0:
            return m.is_charging[e, t - 1] >= m.is_charging[e, t]
        else:
            return Constraint.Skip

    m.charge_immediate = Constraint(m.E, m.T, rule=charge_immediate_rule)

    def discharge_immediate_rule(m, e, t):
        if (t > m.arr_hour[e]) and (m.arr_day[e] == date) and t > 0:
            return m.is_discharging[e, t - 1] >= m.is_discharging[e, t]
        elif m.arr_day[e] < date and t > 0:
            return m.is_discharging[e, t - 1] >= m.is_discharging[e, t]
        else:
            return Constraint.Skip

    m.discharge_immediate = Constraint(m.E, m.T, rule=discharge_immediate_rule)
    # def plugin_immediate_rule(m,e,t):
    #     if (t>m.arr_hour[e]) and (m.arr_day[e] == date):
    #         return (m.is_charging[e,m.arr_hour[e]]+m.is_discharging[e,m.arr_hour[e]])>=(m.is_charging[e,t]+m.is_discharging[e,t])
    #     elif m.arr_day[e]<date:
    #         return (m.is_charging[e,0]+m.is_discharging[e,0])>=(m.is_charging[e,t]+m.is_discharging[e,t])
    #     else:
    #         return Constraint.Skip
    # m.plugin_immediate = Constraint(m.E,m.T,rule=plugin_immediate_rule)

    '''
    SoE non-negative and prepare for future trips constraints
    '''

    # update SoE for event e at hour t
    def SoE_update_rule(m, e, t):
        if t == 0:
            return m.SoE_2d[e, t]
        else:
            return m.SoE_update[e, t - 1] + (m.charge_power[e, t - 1] - m.discharge_power[e, t - 1]) * m.parking_time[
                e, t - 1]

    m.SoE_update = Expression(m.E, m.T, rule=SoE_update_rule)

    # Prepare for next trip:
    def next_trip_min_SoE_rule(m, e, t):
        if (m.park_end_day[e] == date) and (t == m.park_end_hour[e]):
            return m.SoE_update[e, t] >= m.next_trip_e[e]
        else:
            return Constraint.Skip

    m.next_trip_min_SoE = Constraint(m.E, m.T, rule=next_trip_min_SoE_rule)

    # SoE shouldn't be negative:
    def SoE_nonnegative_rule(m, e, t):
        return m.SoE_update[e, t] >= 0

    m.SoE_nonnegative = Constraint(m.E, m.T, rule=SoE_nonnegative_rule)

    # SoE does not exceed battery capacity
    def SoE_le_capacity_rule(m, e, t):
        return m.SoE_update[e, t] <= m.capacity[e]

    m.SoE_le_capacity = Constraint(m.E, m.T, rule=SoE_le_capacity_rule)

    '''
    Avoid alternating charging direction
    '''

    def detect_charge_jump_1(m, e, t):
        if t == 0:
            return m.charge_jump[e, t] == 0
        else:
            return m.is_charging[e, t] - m.is_charging[e, t - 1] <= 100000 * m.charge_jump[e, t]

    m.detect_charge_jump_1 = Constraint(m.E, m.T, rule=detect_charge_jump_1)

    def detect_charge_jump_2(m, e, t):
        if t == 0:
            return m.charge_jump[e, t] == 0
        else:
            return m.is_charging[e, t - 1] - m.is_charging[e, t] >= -100000 * m.charge_jump[e, t]

    m.detect_charge_jump_2 = Constraint(m.E, m.T, rule=detect_charge_jump_2)

    def charge_jump_cnt(m, e):
        return sum(m.charge_jump[e, t] for t in m.T)

    m.charge_jump_cnt = Expression(m.E, rule=charge_jump_cnt)

    def charge_jump_rule(m, e):
        return m.charge_jump_cnt[e] <= 2

    m.charge_jump_rule = Constraint(m.E, rule=charge_jump_rule)

    def detect_discharge_jump_1(m, e, t):
        if t == 0:
            return m.discharge_jump[e, t] == 0
        else:
            return m.is_discharging[e, t] - m.is_discharging[e, t - 1] <= 100000 * m.discharge_jump[e, t]

    m.detect_discharge_jump_1 = Constraint(m.E, m.T, rule=detect_discharge_jump_1)

    def detect_discharge_jump_2(m, e, t):
        if t == 0:
            return m.discharge_jump[e, t] == 0
        else:
            return m.is_discharging[e, t - 1] - m.is_discharging[e, t] >= -100000 * m.discharge_jump[e, t]

    m.detect_discharge_jump_2 = Constraint(m.E, m.T, rule=detect_discharge_jump_2)

    def discharge_jump_cnt(m, e):
        return sum(m.discharge_jump[e, t] for t in m.T)

    m.discharge_jump_cnt = Expression(m.E, rule=discharge_jump_cnt)

    def discharge_jump_rule(m, e):
        return m.charge_jump_cnt[e] <= 2

    m.discharge_jump_rule = Constraint(m.E, rule=discharge_jump_rule)

    def avoid_adj_alternating_rule(m, e, t):
        if t > 0:
            return (m.is_charging[e, t - 1] - m.is_discharging[e, t - 1]) * (
                    m.is_charging[e, t] - m.is_discharging[e, t]) >= 0
        else:
            return Constraint.Skip

    m.avoid_adj_alternating = Constraint(m.E, m.T, rule=avoid_adj_alternating_rule)

    '''
    Min duration of charging/discharging
    '''

    def min_charge_time_rule(m, e):
        return sum(m.is_charging[e, t] * m.parking_time[e, t] + m.is_discharging[e, t] * m.parking_time[e, t] for t in
                   m.T) >= 0.5 * m.is_active[e]

    def activity_rule(m, e):
        M = 1000  # Example of a large constant, assuming this is larger than any possible sum of times
        return sum(m.is_charging[e, t] * m.parking_time[e, t] + m.is_discharging[e, t] * m.parking_time[e, t] for t in
                   m.T) <= M * m.is_active[e]

    m.min_charge_energy = Constraint(m.E, rule=min_charge_time_rule)
    m.activity_constraint = Constraint(m.E, rule=activity_rule)

    '''
    Net Energy Matching
    '''

    # houlry charge energy requested from the grid for all EVs of shifted profile
    def hourly_tot_net_charge_energy_rule(m, t):
        return sum((m.charge_power[e, t] - m.discharge_power[e, t]) * m.parking_time[e, t] for e in m.E)

    m.hourly_tot_net_charge_energy = Expression(m.T, rule=hourly_tot_net_charge_energy_rule)

    # Net charged energy for whole day of shifted profile
    def net_charge_daily_rule(m):
        return sum(m.hourly_tot_net_charge_energy[t] for t in m.T)

    m.shifted_daily_energy = Expression(rule=net_charge_daily_rule)

    # Match shifted daily required energy with the unshifted sum
    def match_daily_energy_rule(m):
        return m.shifted_daily_energy + m.energy_deficit == m.unshifted_daily_energy + m.energy_surplus

    m.match_daily_energy = Constraint(rule=match_daily_energy_rule)

    '''
    Expression for hourly aggregated power
    '''

    # hourly charge power requested from the grid for all EVs
    def hourly_tot_charge_power_rule(m, t):
        return sum(m.charge_power[e, t] * m.is_charging[e, t] for e in m.E)

    m.hourly_tot_charge_power = Expression(m.T, rule=hourly_tot_charge_power_rule)

    def mask_tot_charge_z(m, t):
        return m.tot_charge_z[t] == m.hourly_tot_charge_power[t]

    m.tot_charge = Constraint(m.T, rule=mask_tot_charge_z)

    def hourly_tot_discharge_power_rule(m, t):
        return sum(m.discharge_power[e, t] * m.is_discharging[e, t] for e in m.E)

    m.hourly_tot_discharge_power = Expression(m.T, rule=hourly_tot_discharge_power_rule)

    def mask_tot_discharge_z(m, t):
        return m.tot_discharge_z[t] == m.hourly_tot_discharge_power[t]

    m.tot_discharge = Constraint(m.T, rule=mask_tot_discharge_z)

    def hourly_tot_net_charge_power_rule(m, t):
        return sum((m.charge_power[e, t] - m.discharge_power[e, t]) for e in m.E)

    m.hourly_tot_net_charge_power = Expression(m.T, rule=hourly_tot_net_charge_power_rule)

    ###########################
    # Objective
    def objective_rule(m):
        return sum((m.charge_to_match[t] - m.tot_charge_z[t]) ** 2 for t in m.T) + sum(
            (m.discharge_to_match[t] - m.tot_discharge_z[t]) ** 2 for t in m.T) + sum(
            1000 * (m.charge_power_limit_s[e, t] + m.discharge_power_limit_s[e, t]) for e in m.E for t in
            m.T)  # + sum(m.change_direction[e] for e in m.E)
        # sum((m.charge_to_match[t]-m.hourly_tot_charge_power[t])**2 for t in m.T) + sum((m.discharge_to_match[t]-m.hourly_tot_discharge_power[t])**2 for t in m.T)+ sum(1000*(m.charge_power_limit_s[e,t]+m.discharge_power_limit_s[e,t]) for e in m.E for t in m.T)

    m.objective = Objective(rule=objective_rule, sense=minimize)
    ###########################
    # Solve model
    solver = SolverFactory('gurobi')
    # solver.options['tol'] = 0.0001
    solver.solve(m, tee=True)  # ,keepfiles=True,logfile="match_profile_log.log")
    # Save results
    ch_dict = {(e, t): m.charge_power[e, t].value * emob_max_p for e in m.E for t in
               m.T}  # Denormalize back to normal power value in kW
    dis_dict = {(e, t): m.discharge_power[e, t].value * emob_max_p for e in m.E for t in m.T}
    ch = pd.Series(ch_dict).unstack()
    ch.to_csv(f'{path}/opt_res/369_0_cluster_{cluster_i}_charge.csv')
    dis = pd.Series(dis_dict).unstack()
    dis.to_csv(f'{path}/opt_res/369_0_cluster_{cluster_i}_discharge.csv')
    return 0


# Normalize the mobility aggregated charge power, calculate min, max, sum of power if there were EVs charging in the cluster then optimize this cluster's charging behavior to fit the curve
for i in range(k):

    cluster = clustered[clustered['cluster'] == i]
    emob_agg_e = [sum(x) for x in zip(*cluster['charge_energy_list'])]  # energy in kWh
    emob_agg_p = [sum(x) for x in zip(*cluster['charge_power_list'])]  # power in kW

    emob_max_p, emob_min_p, emob_tot_p = max(emob_agg_p), min(emob_agg_p), sum(emob_agg_p)
    print("Mobility data peak power:", emob_max_p, "Mobility data minimal power:", emob_min_p,
          "Mobility data total power:", emob_tot_p)

    emob_max_e, emob_min_e, emob_tot_e = max(emob_agg_e), min(emob_agg_e), sum(emob_agg_e)
    print("Mobility data peak energy:", emob_max_e, "Mobility data minimal energy:", emob_min_e,
          "Mobility data total energy:", emob_tot_e)

    if emob_max_p != 0:
        emob_agg_p_norm = [p / emob_max_p for p in emob_agg_p]
        # plt.figure(figsize=(10, 6))
        # plt.plot(emob_agg_p, label='power')
        # plt.plot(emob_agg_e, label='energy')
        # plt.legend()
        # plt.savefig(f'{path}/369_9_cluster_{i}_agg_p_e.png')
        cluster.loc[:, 'normalized_chg_power'] = cluster['chg rate'] / emob_max_p
        normalized_tot_e = emob_tot_e / emob_max_p
        opt_res_code = opt_charge_profile(charge, discharge, net_normalized, cluster, emob_max_p, normalized_tot_e,
                                          date, i, path)

    else:
        index = list(cluster['event_index'])
        T = list(range(24))
        res_dict = {(e, t): 0 for e in index for t in T}  # Denormalize back to normal power value in kW
        # Convert the dictionary into a multi-index series to facilitate unstacking
        res = pd.Series(res_dict).unstack()
        # Save the restructured data to CSV
        res.to_csv(f'{path}/opt_res/369_0_cluster_{i}.csv')
