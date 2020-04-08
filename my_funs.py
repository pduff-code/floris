import aep_calculation
import os
import sys
import numpy as np
import pandas as pd
from time import perf_counter
import matplotlib.pyplot as plt
import math
from math import ceil


# Function for testing multiprocessing
def mult(x, y=2):
    return x * y

# function for the aep calcs
def single_computation(lat, lon, wtcap, pcap, spacing, turb_fold_path):
    n_wt= ceil(pcap / wtcap)
    print(pcap, 'MW plant with (', n_wt, ')', wtcap, 'MW turbines')
    aep = aep_calculation.aep_calc()
    aep_GWh, wake_free_aep_GWh, wake_loss_decimal = aep.single_aep_value(lat, 
                                                                         lon, 
                                                                         wtcap, 
                                                                         n_wt, 
                                                                         plot_layout=False, 
                                                                         wake_model='gauss', 
                                                                         fwrop=None, 
                                                                         plot_rose=False,
                                                                         grid_spc=spacing,
                                                                         turb_fold_path=turb_fold_path)
    CF1=8760*n_wt*wtcap # MWh
    #return values in MWh or [-]
    return aep_GWh*1000, wake_free_aep_GWh*1000, wake_loss_decimal, CF1

