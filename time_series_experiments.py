# Patrick Duffy
# NREL 2019

import pandas as pd
import numpy as np
from floris.tools import time_series_utilities as tsu

raw_data = pd.read_csv('Humboldt_2017.csv')

ws_100 = raw_data['windspeed_100m']
time = raw_data['time_index']

ts = tsu.Time_Series(ws_100)

minimum, maximum, mean, std = ts.compute_stats()
print(mean, std)


