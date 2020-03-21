# Make pickled wind rose objects directly on Eagle using 2020 Offshore CA data
import h5py
import numpy as np
import pandas as pd
import glob
import pickle

# folder with raw data for 1 year dataset
fo_path = '/projects/oswwra/tap/region-offshoreCA/WPS1/h5_new/WPS1-WRF7*.h5'
# folder with raw data for 20 year dataset
# fo_path = '/projects/oswwra/tap/prod-offshoreCA/WPS1/h5/Offshore_CA_WTK1_*.h5'


def main():
    # dictionary of sites (Currently Humboldt)
    # TODO make the script read this data in from a csv or txt file
    sites = {'latitude':[40.99776],
            'longitude':[-124.670844],
            'height':[106, 118, 128, 138, 150, 161, 169]} # m

    # flag to save individual .csv files with the data for each coordinate pair
    save_csv = False          

    for i in range(len(sites['latitude'])):
        # initialize the site dataframe
        site_data=pd.DataFrame(columns=['time_index','winddirection_100m', 'winddirection_10m', 'winddirection_120m',
                                    'winddirection_140m', 'winddirection_160m', 'winddirection_200m', 'winddirection_40m', 
                                    'winddirection_60m', 'winddirection_80m', 'windspeed_100m', 'windspeed_10m', 'windspeed_120m', 
                                    'windspeed_140m', 'windspeed_160m', 'windspeed_200m', 'windspeed_40m', 'windspeed_60m', 'windspeed_80m'])

        # get the wind resource for the coordinate
        lat = sites['latitude'][i]
        lon = sites['longitude'][i]

        # Open all .h5 files in folder
        for filename in glob.glob(fo_path): 
            f = h5py.File(filename)
            
            # find the index
            coord_data = f['coordinates']
            lat_list = coord_data[:,0]
            long_list = coord_data[:,1]
            lat_diff = abs(lat_list-lat)
            long_diff= abs(long_list-lon)
            combined_min = lat_diff+long_diff
            index = np.argmin(combined_min)
        
            # grab the data from the indices of interest and concatenate
            time_series = pd.DataFrame(columns=['time_index','winddirection_100m', 'winddirection_10m', 'winddirection_120m',
                                    'winddirection_140m', 'winddirection_160m', 'winddirection_200m', 'winddirection_40m', 
                                    'winddirection_60m', 'winddirection_80m', 'windspeed_100m', 'windspeed_10m', 'windspeed_120m', 
                                    'windspeed_140m', 'windspeed_160m', 'windspeed_200m', 'windspeed_40m', 'windspeed_60m', 'windspeed_80m'])
            for var in ['time_index','winddirection_100m', 'winddirection_10m', 'winddirection_120m', 
                        'winddirection_140m', 'winddirection_160m', 'winddirection_200m', 'winddirection_40m', 
                        'winddirection_60m', 'winddirection_80m', 'windspeed_100m', 'windspeed_10m', 'windspeed_120m', 
                        'windspeed_140m', 'windspeed_160m', 'windspeed_200m', 'windspeed_40m', 'windspeed_60m', 'windspeed_80m']:
                ds = f[var]
                if var == 'time_index':
                    time_series[var] = ds
                else:
                    scale_factor = ds.attrs['scale_factor']
                    time_series[var] = ds[:, index] / scale_factor
            
            site_data=site_data.append(time_series, ignore_index=True)
            # TODO could double check the order of glob.glob if interested in the time series

        # save the dataframe for the site
        if save_csv:
            site_data.to_csv(sites['name'][i]+ '.csv', header=True)
        
        # make the pickled python file
        wd=np.arange(0, 360, 5.)
        ws=np.arange(0, 26, 1.)
        d=site_data

        # Array of hub height data avaliable from data
        h_range = [10, 40, 60, 80, 100, 120, 140, 160, 200]

        # make pickle files at the different hub heights
        for ht in sites ['height']:

            # Check inputs       
            if (h_range[0] > ht):
                print('Error, height is not in the range of avaliable data. Minimum height = 10m')

            if (h_range[-1] < ht):
                print('Error, height is not in the range of avaliable data. Maxiumum height = 200m')
                            
            # Case for turbine height (ht) matching discrete avaliable height (h_range) 
            if ht in h_range:

                # grab from one set 
                ws_new = d['windspeed_'+str(ht)+'m']
                wd_new = d['winddirection_'+str(ht)+'m']
                
            # Case for ht not matching discete height
            else: 
                h_range_up = next(x[0] for x in enumerate(h_range) if x[1] > ht)
                h_range_low = h_range_up - 1
                hub_up = h_range[h_range_up]
                hub_low = h_range[h_range_low]
                
                # Wind Speed interpolation
                ws_low = d['windspeed_'+str(hub_low)+'m']
                ws_high = d['windspeed_'+str(hub_up)+'m']
                
                ws_new = np.array(ws_low) * (1-((ht - hub_low)/(hub_up - hub_low))) \
                    + np.array(ws_high) * ((ht - hub_low)/(hub_up - hub_low))
                
                # Wind Direction interpolation using Circular Mean method 
                wd_low = d['winddirection_'+str(hub_low)+'m']
                wd_high = d['winddirection_'+str(hub_up)+'m']

                sin0 = np.sin(np.array(wd_low) * (np.pi/180))
                cos0 = np.cos(np.array(wd_low) * (np.pi/180))
                sin1= np.sin(np.array(wd_high) * (np.pi/180))
                cos1 = np.cos(np.array(wd_high) * (np.pi/180))

                sin_wd = sin0 * (1-((ht - hub_low)/(hub_up - hub_low)))+ sin1 * \
                    ((ht - hub_low)/(hub_up - hub_low))
                cos_wd = cos0 * (1-((ht - hub_low)/(hub_up - hub_low)))+ cos1 * \
                    ((ht - hub_low)/(hub_up - hub_low))
                    
                # Interpolated wind direction 
                wd_new = 180/np.pi * np.arctan2(sin_wd, cos_wd)
            
            # Create a dataframe named df
            df= pd.DataFrame({'ws': ws_new,
                                'wd': wd_new})
                    
            # Start by simply round and wrapping the wind direction and wind speed columns
            df['wd'] = wrap_360(df.wd.round())
            df['ws'] = wrap_360(df.ws.round())
            
            # Now group up
            df['freq_val'] = 1.
            df = df.groupby(['ws', 'wd']).sum()
            df['freq_val'] = df.freq_val.astype(float) / df.freq_val.sum()
            df = df.reset_index()
            
            # Resample onto the provided wind speed and wind direction binnings
            num_ws = len(ws)
            num_wd = len(wd)
            ws_step = ws[1] - ws[0]
            wd_step = wd[1] - wd[0]
            df = resample_wind_speed(df, ws=ws)
            df = resample_wind_direction(df, wd=wd)
            
            # name and save the pickled python file
            latstr = str(lat)
            latstr = latstr.replace('.','_')
            lonstr = str(lon)
            lonstr = lonstr.replace('.','_')
            filename = str(int(ht))+'mlat'+latstr+'long'+lonstr+'wtk.p'
            
            pickle.dump([num_wd, num_ws, wd_step, ws_step, wd, ws, df], open(filename, "wb"))

#########################################################################    
def resample_wind_speed(df, ws=np.arange(0, 26, 1.)):
        """
        Modify the default bins for sorting wind speed.

        Args:
            df (pd.DataFrame): Wind speed data
            ws (np.array, optional): Vector of wind speed bins for
                WindRose. Defaults to np.arange(0, 26, 1.).

        Returns:
            df (pd.DataFrame): Resampled wind speed for WindRose
        """
        # Make a copy of incoming dataframe
        df = df.copy(deep=True)

        # Get the wind step
        ws_step = ws[1] - ws[0]

        # Ws
        ws_edges = (ws - ws_step / 2.0)
        ws_edges = np.append(ws_edges, np.array(ws[-1] + ws_step / 2.0))

        # Cut wind speed onto bins
        df['ws'] = pd.cut(df.ws, ws_edges, labels=ws)

        # Regroup
        df = df.groupby(['ws', 'wd']).sum()

        # Fill nans
        df = df.fillna(0)

        # Reset the index
        df = df.reset_index()

        # Set to float
        df['ws'] = df.ws.astype(float)
        df['wd'] = df.wd.astype(float)

        return df


def resample_wind_direction(df, wd=np.arange(0, 360, 5.)):
        """
        Modify the default bins for sorting wind direction.

        Args:
            df (pd.DataFrame): Wind direction data
                wd (np.array, optional): Vector of wind direction bins
                for WindRose. Defaults to np.arange(0, 360, 5.).

        Returns:
            df (pd.DataFrame): Resampled wind direction for WindRose
        """
        # Make a copy of incoming dataframe
        df = df.copy(deep=True)

        # Get the wind step
        wd_step = wd[1] - wd[0]

        # Get bin edges
        wd_edges = (wd - wd_step / 2.0)
        wd_edges = np.append(wd_edges, np.array(wd[-1] + wd_step / 2.0))

        # Get the overhangs
        negative_overhang = wd_edges[0]
        positive_overhang = wd_edges[-1] - 360.

        # Need potentially to wrap high angle direction to negative for correct binning
        df['wd'] = wrap_360(df.wd)
        if negative_overhang < 0:
            print('Correcting negative Overhang:%.1f' % negative_overhang)
            df['wd'] = np.where(df.wd.values >= 360. + negative_overhang,
                                df.wd.values - 360., df.wd.values)

        # Check on other side
        if positive_overhang > 0:
            print('Correcting positive Overhang:%.1f' % positive_overhang)
            df['wd'] = np.where(df.wd.values <= positive_overhang,
                                df.wd.values + 360., df.wd.values)

        # Cut into bins
        df['wd'] = pd.cut(df.wd, wd_edges, labels=wd)

        # Regroup
        df = df.groupby(['ws', 'wd']).sum()

        # Fill nans
        df = df.fillna(0)

        # Reset the index
        df = df.reset_index()

        # Set to float Re-wrap
        df['wd'] = df.wd.astype(float)
        df['ws'] = df.ws.astype(float)
        df['wd'] = wrap_360(df.wd)

        return df

def wrap_360(x):
    """
    Wrap an angle to between 0 and 360

    Returns:
        [array]: angles in specified interval
    """
    x = np.where(x < 0., x + 360., x)
    x = np.where(x >= 360., x - 360., x)
    return (x)

#############################################################
if __name__ == "__main__":
    main()