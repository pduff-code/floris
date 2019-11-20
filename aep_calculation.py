# NREL 2019
# Patrick Duffy

# Imports
import matplotlib.pyplot as plt
import floris.tools as wfct
import floris.tools.visualization as vis
from floris.tools.optimization import YawOptimizationWindRose
import floris.tools.wind_rose as rose
import floris.tools.power_rose as pr
import numpy as np
import pandas as pd


""" misc. notes
Quick and dirty AEP calculation module
"""

class aep_calc():
    
    #def __init__(self, default inputs):
        

    def single_aep_value(self, lat, lon, plot_layout, fwrop, plot_rose):
        
        # setup floris object
        fi = wfct.floris_utilities.FlorisInterface('examples\example_input.json')
        
        # Set wind farm to N_row x N_row grid with constant spacing 
        D = fi.floris.farm.turbines[0].rotor_diameter
        N_row = 5
        spc = 7
        layout_x = []
        layout_y = []
        for i in range(N_row):
        	for k in range(N_row):
        		layout_x.append(i*spc*D)
        		layout_y.append(k*spc*D)
        N_turb = len(layout_x)
        
        fi.reinitialize_flow_field(layout_array=(layout_x, layout_y),wind_direction=270.0,wind_speed=8.0)
        fi.calculate_wake() # must do this t 
        
         
        # plot wf layout
        if plot_layout:
            fig, ax = plt.subplots()
            vis.plot_turbines(ax, layout_x, layout_y, yaw_angles=np.zeros(len(layout_x)), D=D)
            ax.set_title('Wind Farm Layout')
        
        # confirm that I don't need this
           
        # set min and max yaw offsets for optimization
        min_yaw = 0.0
        max_yaw = 0.0 # 0 for no yaw optimization
        
        # Define minimum and maximum wind speed for optimizing power. 
        # Below minimum wind speed, assumes power is zero.
        # Above maximum_ws, assume optimal yaw offsets are 0 degrees
        minimum_ws = 4.0
        maximum_ws = 25.0
        
        
        # wind data (either from wind rose object, or from windtoolkit)
        wind_rose = rose.WindRose()
        if fwrop is None:
            print('Accessing WindToolkit')
            # fetch the data from wind tookit
            wd_list = np.arange(0,360,5)
            ws_list = np.arange(0,26,1)

            df = wind_rose.import_from_wind_toolkit_hsds(lat,
                                                         lon,
                                                         ht = 100,
                                                         wd = wd_list,
                                                         ws = ws_list,
                                                         limit_month = None,
                                                         st_date = None,
                                                         en_date = None)
        else:
            df = wind_rose.load(fwrop)
        
        if plot_rose:
            wind_rose.plot_wind_rose()
            
        # Instantiate the Optimization object
        yaw_opt = YawOptimizationWindRose(fi, df.wd, df.ws,
                                       minimum_yaw_angle=min_yaw,
                                       maximum_yaw_angle=max_yaw,
                                       minimum_ws=minimum_ws,
                                       maximum_ws=maximum_ws)
        
        # Determine baseline power with and without wakes
        df_base = yaw_opt.calc_baseline_power()
        
        
        # combine wind farm-level power into one dataframe
        df_power = pd.DataFrame({'ws':df.ws,'wd':df.wd, \
            'freq_val':df.freq_val,'power_no_wake':df_base.power_no_wake, \
            'power_baseline':df_base.power_baseline})
        
        # Set up the power rose
        df_turbine_power_no_wake = pd.DataFrame([list(row) for row in df_base['turbine_power_no_wake']],columns=[str(i) for i in range(1,N_turb+1)])
        df_turbine_power_no_wake['ws'] = df.ws
        df_turbine_power_no_wake['wd'] = df.wd
        df_turbine_power_baseline = pd.DataFrame([list(row) for row in df_base['turbine_power_baseline']],columns=[str(i) for i in range(1,N_turb+1)])
        df_turbine_power_baseline['ws'] = df.ws
        df_turbine_power_baseline['wd'] = df.wd
        
        case_name = 'Example '+str(N_row)+' x '+str(N_row)+ ' Wind Farm'
        power_rose = pr.PowerRose(case_name, df_power, df_turbine_power_no_wake, df_turbine_power_baseline)
        
        # Display AEP analysis
        fig, axarr = plt.subplots(2, 1, sharex=True, figsize=(6.4, 6.5))
        power_rose.plot_by_direction(axarr)
        power_rose.report()
        
        plt.show()  
        
        
def main():
    # run the aep calculation via the aep class
    
    # these are for center usa
    #lat = 39.8283
    #lon = -98.5795
    #wind_data = 'windtoolkit_geo_center_us.p'
    
    wind_data = None
    plot_layout = True
    plot_rose = True

    # california offshore
    lat = 35.236497
    lon= -120.991624
   
    aep=aep_calc()
    #aep.single_aep_value(lat,lon,plot_layout,wind_data,plot_rose)
    aep.single_aep_value(lat, lon, plot_layout, wind_data, plot_rose)
    
    
    
    
main()
    