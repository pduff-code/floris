# Patrick Duffy
# NREL 2019
# Much of the script based on the DTU course:
# Introduction to Micrometeorology for Wind Energy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Time_Series():
    """
    Base class for a time series and analysis tools.
    """

    def __init__(self, ts, t=None, dt=1):
        """
        Initialization of time_series object.

        Args:
        ------
        ts : float
            the time series values
        t  : float, int, or str
            time_stamps (optional) default to None
        dt : float or int
            time step size delta t (optional) default to unit step
        """
        self.ts = ts
        self.dt = dt
        self.t = t

    def compute_stats(self):
        """ 
        Computes and returns min, max, mean, and std
        """
        self.minimum = min(self.ts)
        self.maximum = max(self.ts)
        self.mean = np.mean(self.ts)
        self.std = np.std(self.ts)

        return self.minimum, self.maximum, self.mean, self.std

    def inspect(self):
        """
        Plot the time series for inspection.
        """
        plt.figure()
        plt.plot(self.t, self.ts)
        plt.xlabel('t')
        plt.ylabel('Time series')
        plt.show()


class Wind_Time_Series(Time_Series):
    """ 
    Wind specific subclass of Time_Series.
    (TODO: Finish writing the methods for the subclass)
    """

    def __init__(self, ts, t=None, dt=1):
        """
        Wind specific methods for:
        Calculating extreme value (for some return period)
        Computing a weibull A and K for the time series
        Computing and plotting a power spectrum
        """
        super().__init__(ts, t=None, dt=1)


    def extreme_value(self, method, return_period=None):
        """
        Return extreme value based on gumbel method
        add other methods ( could even have different options)
        """

        if method == 'naive':
            # do stuff
            pass
        elif method == 'gumbel':
            # do different stuff
            pass
        elif method == 'ordered_statistics':
            # do even more different stuff
            pass
        else:
            print('Please specify an extreme value method: gumbel, ordered statistics, or naive.')

    def get_weibull_ak(self):
        """
        Return weibull a and k based on a method
        """
        pass

    def power_spectrum(self):
        """
        Compute and plot the power spectrum
        """
        pass
    

    