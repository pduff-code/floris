# Patrick Duffy
# NREL 2020
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import dates as mpl_dates
import floris.tools as wfct
import floris.tools.time_series_utilities as tsu
from floris.tools.optimization import YawOptimizationWindRose


# Instantiate the FLORIS object
fi = wfct.floris_utilities.FlorisInterface("examples/example_input.json")

# Set wind farm to N_row x N_row grid with constant spacing 
# (2 x 2 grid, 5 D spacing)
D = fi.floris.farm.turbines[0].rotor_diameter
N_row = 2
spc = 5
layout_x = []
layout_y = []
for i in range(N_row):
	for k in range(N_row):
		layout_x.append(i*spc*D)
		layout_y.append(k*spc*D)
N_turb = len(layout_x)
fi.reinitialize_flow_field(layout_array=(layout_x, layout_y),wind_direction=270.0,wind_speed=8.0)
fi.calculate_wake()

# Import wind speed data 
raw_data = pd.read_csv('Humboldt_2017.csv')
ws_100 = raw_data['windspeed_100m'][:2016]
wd_100 = raw_data['winddirection_100m'][:2016]
raw_time = raw_data['time_index'][:2016]

# Clean up time string to datetime
time = []
for ii in range(len(raw_time)):
    t = raw_time[ii]
    s = t[2:-1]
    ts = datetime.strptime(s, '%Y-%m-%d %X')
    time.append(ts)

# Import some electricity data
elec_data = pd.read_csv('California_region_electricity_demand.csv', usecols=range(5))
elec_data['time'] = pd.to_datetime(elec_data['time'], infer_datetime_format=True)

# Using the optimization framework for the computations
yaw_opt = YawOptimizationWindRose(fi, wd_100, ws_100,
                               minimum_yaw_angle=0.0,
                               maximum_yaw_angle=0.0,
                               minimum_ws= min(ws_100),
                               maximum_ws= max(ws_100))

df_base = yaw_opt.calc_baseline_power()
df_base['time'] = time

baseline = np.divide(df_base['power_baseline'], (10e6) ) # MW
print(df_base['power_baseline'][1997])
print(baseline[1997])
# Make a Plot
fig, ax = plt.subplots()
ax.plot(df_base['time'], baseline)
fig.autofmt_xdate() # rotate and align the tick labels so they look better
ax.fmt_xdata = mpl_dates.DateFormatter('%Y-%m-%d')
ax.set_title('First week of 2017')
ax.set_ylabel('Wind Farm Power [MW]')
plt.show()

# Make an overlay plot
fig, ax1 = plt.subplots()
fig.autofmt_xdate() # rotate and align the tick labels so they look better
ax1.set_ylabel('Wind Farm Power [MW]',color='r')
ax1.plot(df_base['time'], baseline,color='r')
ax1.tick_params(axis='y', labelcolor='r')
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'b'
ax2.set_ylabel('Elec Demand in CA [MWh]', color=color) # unit is in MWh
ax2.plot(elec_data['time'], elec_data['Demand'], color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


# leave it here for now
# TODO be able to ensamble average over different intervals and plot the 'avg day in march' for example