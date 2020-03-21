# NREL 2019
# Patrick Duffy

import matplotlib.pyplot as plt
import floris.tools as wfct
import floris.tools.visualization as vis
from floris.tools.optimization_pd import YawOptimizationWindRose
import floris.tools.wind_rose as rose
import floris.tools.power_rose as pr
import numpy as np
import pandas as pd
from math import sqrt, floor
import os


class aep_calc():
    
    def __init__(self):
        """Initialize AEP calculation
        
        AEP calculation class. A container for the single_aep_value() method.
        
        """
        
    def single_aep_value(self, lat, lon, wt_cap_MW, n_wt, plot_layout=False, 
                         wake_model='gauss', fwrop=None, plot_rose=False,
                         grid_spc=7, wind_resource='old', turb_fold_path=None):
        """Single AEP value
        
        Calculates AEP for single set of input parameters. Returns waked and 
        wake-free AEP, as well as the percent wake loss.
        
        Inputs
        ----------
        lat : latitude coordinate in decimal degrees
        lon : longitude coordinate in decimal degrees
        wt_cap_MW  : Individual wind turbine capacity in MW
        n_wt : Number of wind turbines
        plot_layout : Plots the wind farm layout when true. Default: False
        wake_model : Wake model used in the calculation. Default: gauss
        fwrop : Floris wind rose object path. Default: None (uses Wind Toolkit)
        plot_rose : Plots wind rose used in the AEP calculation. Default: False
        grd_spc : turbine spacing in grid layout in # rotor diams. Default: 7
        
        (Still want to implement these as inputs):
            wts : wind turbines in the wind farm, and their respective properties
                want to check a library for a given rated power value and load
                if a turbine model is not in the library, raise an exception
            wake_combin : wake summation method for multiple wakes
    
        Returns
        -------
        wake_free_aep_GWh : AEP ignoring wake effects in GWh
        aep_GWh : AEP including wake effects in GWh
        percent_loss : Percent of wake_free_aep lost due to wakes
        """
        
        data_file_name = str(int(wt_cap_MW))+'MW.json'
        if  turb_fold_path is None:
            file_path = 'C:/Users/pduffy/Documents/repos/orca/ORCA/ORCA/library/turbines/ATB/PBturbines/'+data_file_name
        else:
            file_path = turb_fold_path+data_file_name
        
        print(file_path)
        if os.path.isfile(file_path):
            fi = wfct.floris_utilities.FlorisInterface(file_path)
        
        # Make a grid layout
        D = fi.floris.farm.turbines[0].rotor_diameter
        print(D)
        h_hub = fi.floris.farm.turbines[0].hub_height
        print(grid_spc)
        layout_x, layout_y =self.make_grid_layout(n_wt, 
                                                  D,
                                                  grid_spc)

        # Set up the model and update the floris object
        fi.floris.farm.set_wake_model(wake_model)
        fi.reinitialize_flow_field(layout_array=(layout_x, layout_y),
                                   wind_direction=0.0,wind_speed=8.0)
        fi.calculate_wake()
        
        # Plot wf layout
        if plot_layout:
            fig, ax = plt.subplots()
            vis.plot_turbines(ax, layout_x, layout_y, 
                              yaw_angles=np.zeros(len(layout_x)), D=D)
            ax.set_title('Wind Farm Layout')
        
        # Wind resource data filename
        latstr = str(lat)
        latstr = latstr.replace('.','_')
        lonstr = str(lon)
        lonstr = lonstr.replace('.','_')
        hstr = str(h_hub)
        hstr = hstr.replace('.','_')
        resource_file_name = 'lat'+latstr+'long'+lonstr+'wtk.p'
        resource_file = 'C:/Users/pduffy/Documents/repos/orca/ORCA/ORCA/library/wind_resource/'+wind_resource+'/'+hstr+'m/' +resource_file_name
        print(resource_file)
        # TODO: should add a mkdir-like command if the folder does not exist.   
        
        # Check for existing wind resource objects and which data to use ('old' refers to Wind Toolkit)
        wind_rose = rose.WindRose()
        if os.path.isfile(resource_file):
            df = wind_rose.load(resource_file)
            
        elif wind_resource == 'old':
            # Fetch the data from wind tookit
            print('Accessing Wind Toolkit...')
            wd_list = np.arange(0,360,5) # 5 degree wd bins
            ws_list = np.arange(0,26,1)  # 1 m/s ws bins 

            df = wind_rose.import_from_wind_toolkit_hsds(lat,
                                                         lon,
                                                         ht = h_hub, 
                                                         wd = wd_list,
                                                         ws = ws_list,
                                                         limit_month = None,
                                                         st_date = None,
                                                         en_date = None)
            
            # Save the wind rose object in ORCA/library/wind_resource/old/<height>m/
            wind_rose.save(resource_file)
            
        else:
            raise Exception('No existing FLORIS wind rose object for the new data at this position.')
            
        # Plot a wind rose
        if plot_rose:
            wind_rose.plot_wind_rose()

        # Instantiate the Optimization object
        yaw_opt = YawOptimizationWindRose(fi, df.wd, df.ws,
                                       minimum_yaw_angle=0,
                                       maximum_yaw_angle=0,
                                       minimum_ws=4.0, 
                                       maximum_ws=25.0) 

        # Determine baseline power with and without wakes
        df_base = yaw_opt.calc_baseline_power()


        # If true save power data for use in Lookup Tables
        savedata=False
        
        if savedata:
            lut_file_name = str(n_wt)+'WT_'+str(int(wt_cap_MW))+'MWturb_'+str(grid_spc)+'Dspacing.csv' 
            # TODO: improve naming convention to include: plant rotation, wake model and wake summation method
            
            lut_file = 'C:/Users/pduffy/Documents/repos/orca/ORCA/ORCA/library/power_lookup_tables/ATB/PBturbines/'+lut_file_name
            df_base.to_csv(lut_file, index=True)
        
        # Combine wind farm-level power into one dataframe
        df_power = pd.DataFrame({'ws':df.ws,'wd':df.wd, \
            'freq_val':df.freq_val,'power_no_wake':df_base.power_no_wake, \
            'power_baseline':df_base.power_baseline})

        # Set up the power rose
        df_turbine_power_no_wake = pd.DataFrame([list(row) for row in df_base['turbine_power_no_wake']],
                                                 columns=[str(i) for i in range(1,n_wt+1)])
        df_turbine_power_no_wake['ws'] = df.ws
        df_turbine_power_no_wake['wd'] = df.wd
        df_turbine_power_baseline = pd.DataFrame([list(row) for row in df_base['turbine_power_baseline']],
                                                  columns=[str(i) for i in range(1,n_wt+1)])
        df_turbine_power_baseline['ws'] = df.ws
        df_turbine_power_baseline['wd'] = df.wd
        case_name = 'Wind Farm'
        power_rose = pr.PowerRose(case_name, df_power, df_turbine_power_no_wake,
                                  df_turbine_power_baseline)

        # Values to return
        wake_free_aep_GWh = power_rose.total_no_wake
        aep_GWh = power_rose.total_baseline
        wake_loss_decimal = power_rose.baseline_wake_loss

        return aep_GWh, wake_free_aep_GWh, wake_loss_decimal
      
       
    def isPerfect(self, N): 
        """Function to check if a number is perfect square or not
        
        taken from: 
        https://www.geeksforgeeks.org/closest-perfect-square-and-its-distance/
        by sahishelangia 
        """
        if (sqrt(N) - floor(sqrt(N)) != 0): 
            return False
        return True
      
    
    def getClosestPerfectSquare(self, N):
        """Function to find the closest perfect square taking minimum steps to
            reach from a number  
            
        taken from: 
        https://www.geeksforgeeks.org/closest-perfect-square-and-its-distance/
        by sahishelangia 
        """
        if (self.isPerfect(N)):  
            distance = 0
            return N, distance
      
        # Variables to store first perfect square number above and below N  
        aboveN = -1
        belowN = -1
        n1 = 0
      
        # Finding first perfect square number greater than N  
        n1 = N + 1
        while (True): 
            if (self.isPerfect(n1)): 
                aboveN = n1  
                break
            else: 
                n1 += 1
      
        # Finding first perfect square number less than N  
        n1 = N - 1
        while (True):  
            if (self.isPerfect(n1)):  
                belowN = n1  
                break
            else: 
                n1 -= 1
                  
        # Variables to store the differences  
        diff1 = aboveN - N  
        diff2 = N - belowN  
      
        if (diff1 > diff2): 
            return belowN, -diff2  
        else: 
            return aboveN, diff1
     
    
    def make_grid_layout(self,n_wt, D, grid_spc):
        """Make a grid layout (close as possible to a square grid)
        
        Inputs:
        -------
            wt_cap_MW : float
                Wind turbine capacity in MW
            n_wt : float
                Number of wind turbines in the plant
            D : float (or might want array_like if diff wt models are used)
                Wind turbine rotor diameter(s) in meters
            grid_spc : float
                Spacing between rows and columns in number of rotor diams D
        
        Returns:
        --------
            layout_x : array_like
                X positions of the wind turbines in the plant
            layout_y : array_like
                Y positions of the wind turbines in the plant
        """
        # Initialize layout variables
        layout_x = []
        layout_y = []
        
        # Find the closest square root
        close_square, dist = self.getClosestPerfectSquare(n_wt)
        side_length = int(sqrt(close_square))
        
        # Build a square grid
        for i in range(side_length):
            for k in range(side_length):
                layout_x.append(i*grid_spc*D)
                layout_y.append(k*grid_spc*D)
        
        # Check dist and determine what to do
        if dist == 0:
            # do nothing
            pass
        elif dist > 0:
            # square>n_wt : remove locations
            del(layout_x[close_square-dist:close_square])
            del(layout_y[close_square-dist:close_square])
            # maybe convert format and transpose    
        else:
            # square < n_w_t : add a partial row
            for i in range(abs(dist)):
                layout_x.append(sqrt(close_square)*grid_spc*D)
                layout_y.append(i*grid_spc*D)        
        
        return layout_x, layout_y
    
    
        
def main():
    # location and wind rose data
    # center usa:
    lat = 39.8283
    lon = -98.5795
    wind_data = 'windtoolkit_geo_center_us.p'
    # california offshore:
    #lat = 35.236497
    #lon= -120.991624
    #wind_data = None
    
    wt_cap_MW = 5
    n_wt = 4
    
    # possible wake models: 'curl', 'gauss', 'jensen', 'multizone'
    aep=aep_calc()
    aep_GWh,wake_free_aep_GWh,percent_loss = aep.single_aep_value(lat, 
                                                                  lon, 
                                                                  wt_cap_MW, 
                                                                  n_wt,
                                                                  plot_layout = False,
                                                                  wake_model='jensen',
                                                                  fwrop=wind_data,
                                                                  plot_rose = False)
    print(aep_GWh, wake_free_aep_GWh, percent_loss)

#main()
    