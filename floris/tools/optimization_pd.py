# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
from itertools import repeat

try:
    from mpi4py.futures import MPIPoolExecutor
except ImportError:
    pass

# import warnings
# warnings.simplefilter('ignore', RuntimeWarning)


class Optimization():
    """
    Base optimization class.

    Args:
        fi (:py:class:`floris.tools.floris_utilities.FlorisInterface`): 
            Interface from FLORIS to the tools package.

    Returns:
        Optimization: An instantiated Optimization object.
    """

    def __init__(self, fi):
        """
        Instantiate Optimization object and its parameters.
        """
        self.fi = fi

    # Private methods

    def _reinitialize(self):
        pass

    def _norm(self, val, x1, x2):
        return (val - x1)/(x2 - x1)
    
    def _unnorm(self, val, x1, x2):
        return np.array(val)*(x2 - x1) + x1

    # Properties

    @property
    def nturbs(self):
        """
        This property returns the number of turbines in the FLORIS 
        object.

        Returns:
            nturbs (int): The number of turbines in the FLORIS object.
        """
        self._nturbs = len(self.fi.floris.farm.turbine_map.turbines)
        return self._nturbs

class YawOptimization(Optimization):
    """
    Sub class of the :py:class`floris.tools.optimization.Optimization`
    object class that performs yaw optimization for a single set of 
    inflow conditions.

    Args:
        fi (:py:class:`floris.tools.floris_utilities.FlorisInterface`): 
            Interface from FLORIS to the tools package.
        minimum_yaw_angle (float, optional): Minimum constraint on 
            yaw. Defaults to None.
        maximum_yaw_angle (float, optional): Maximum constraint on 
            yaw. Defaults to None.
        x0 (iterable, optional): The initial yaw conditions. 
            Defaults to None. Initializes to the current turbine 
            yaw settings.
        bnds (iterable, optional): Bounds for the optimization 
            variables (pairs of min/max values for each variable). 
            Defaults to None. Initializes to [(0.0, 25.0)].
        opt_method (str, optional): The optimization method for 
            scipy.optimize.minize to use. Defaults to None. 
            Initializes to 'SLSQP'.
        opt_options (dictionary, optional): Optimization options for 
            scipy.optimize.minize to use. Defaults to None. 
            Initializes to {'maxiter': 100, 'disp': False,
            'iprint': 1, 'ftol': 1e-7, 'eps': 0.01}.
        include_unc (bool): If True, uncertainty in wind direction 
            and/or yaw position is included when determining wind farm power. 
            Uncertainty is included by computing the mean wind farm power for 
            a distribution of wind direction and yaw position deviations from 
            the original wind direction and yaw angles. Defaults to False.
        unc_pmfs (dictionary, optional): A dictionary containing optional 
            probability mass functions describing the distribution of wind 
            direction and yaw position deviations when wind direction and/or 
            yaw position uncertainty is included in the power calculations. 
            Contains the following key-value pairs:  

            -   **wd_unc**: A numpy array containing wind direction deviations 
                from the original wind direction. 
            -   **wd_unc_pmf**: A numpy array containing the probability of 
                each wind direction deviation in **wd_unc** occuring. 
            -   **yaw_unc**: A numpy array containing yaw angle deviations 
                from the original yaw angles. 
            -   **yaw_unc_pmf**: A numpy array containing the probability of 
                each yaw angle deviation in **yaw_unc** occuring.

            Defaults to None, in which case default PMFs are calculated using 
            values provided in **unc_options**.
        unc_options (disctionary, optional): A dictionary containing values used 
            to create normally-distributed, zero-mean probability mass functions 
            describing the distribution of wind direction and yaw position 
            deviations when wind direction and/or yaw position uncertainty is 
            included. This argument is only used when **unc_pmfs** is None and 
            contains the following key-value pairs:

            -   **std_wd**: A float containing the standard deviation of the wind 
                    direction deviations from the original wind direction.
            -   **std_yaw**: A float containing the standard deviation of the yaw 
                    angle deviations from the original yaw angles.
            -   **pmf_res**: A float containing the resolution in degrees of the 
                    wind direction and yaw angle PMFs.
            -   **pdf_cutoff**: A float containing the cumulative distribution 
                function value at which the tails of the PMFs are truncated. 

            Defaults to None. Initializes to {'std_wd': 4.95, 'std_yaw': 1.75, 
            'pmf_res': 1.0, 'pdf_cutoff': 0.995}.

    Returns:
        YawOptimization: An instantiated YawOptimization object.
    """

    def __init__(self, fi, minimum_yaw_angle=0.0,
                           maximum_yaw_angle=25.0,
                           x0=None,
                           bnds=None,
                           opt_method='SLSQP',
                           opt_options=None,
                           include_unc=False,
                           unc_pmfs=None,
                           unc_options=None):
        """
        Instantiate YawOptimization object and parameter values.
        """
        super().__init__(fi)

        if opt_options is None:
            self.opt_options = {'maxiter': 100, 'disp': False, \
                        'iprint': 1, 'ftol': 1e-7, 'eps': 0.01}

        self.unc_pmfs = unc_pmfs

        if unc_options is None:
            self.unc_options = {'std_wd': 4.95, 'std_yaw': 1.75, \
                        'pmf_res': 1.0, 'pdf_cutoff': 0.995}
        
        self.reinitialize_opt(
            minimum_yaw_angle=minimum_yaw_angle,
            maximum_yaw_angle=maximum_yaw_angle,
            x0=x0,
            bnds=bnds,
            opt_method=opt_method,
            opt_options=opt_options,
            include_unc=include_unc,
            unc_pmfs=unc_pmfs,
            unc_options=unc_options
        )

    # Private methods

    def _yaw_power_opt(self, yaw_angles):
        return -1 * self.fi.get_farm_power_for_yaw_angle(yaw_angles, include_unc=self.include_unc, \
            unc_pmfs=self.unc_pmfs, unc_options=self.unc_options)

    def _optimize(self):
        """
        Find optimum setting of turbine yaw angles for power production
        given fixed atmospheric conditins (wind speed, direction, etc.).

        Returns:
            opt_yaw_angles (np.array): optimal yaw angles of each turbine.
        """

        self.residual_plant = minimize(self._yaw_power_opt,
                                self.x0,
                                method=self.opt_method,
                                bounds=self.bnds,
                                options=self.opt_options)

        opt_yaw_angles = self.residual_plant.x

        return opt_yaw_angles

    def _set_opt_bounds(self, minimum_yaw_angle, maximum_yaw_angle):
        self.bnds = [(minimum_yaw_angle, maximum_yaw_angle) for _ in \
                     range(self.nturbs)]

    # Public methods

    def optimize(self):
        """
        Find optimum setting of turbine yaw angles for power production
        given fixed atmospheric conditins (wind speed, direction, etc.).

        Returns:
            opt_yaw_angles (np.array): optimal yaw angles of each turbine.
        """
        print('=====================================================')
        print('Optimizing wake redirection control...')
        print('Number of parameters to optimize = ', len(self.x0))
        print('=====================================================')

        opt_yaw_angles = self._optimize()

        if np.sum(opt_yaw_angles) == 0:
            print('No change in controls suggested for this inflow \
                   condition...')

        return opt_yaw_angles

    def reinitialize_opt(self, minimum_yaw_angle=None,
                           maximum_yaw_angle=None,
                           x0=None,
                           bnds=None,
                           opt_method=None,
                           opt_options=None,
                           include_unc=None,
                           unc_pmfs=None,
                           unc_options=None):
        """
        Reintializes parameter values for the optimization.
        
        This method reinitializes the optimization parameters and 
        bounds to the supplied values or uses what is currently stored.
        
        Args:
            fi (:py:class:`floris.tools.floris_utilities.FlorisInterface`): 
                Interface from FLORIS to the tools package.
            minimum_yaw_angle (float, optional): Minimum constraint on 
                yaw. Defaults to None.
            maximum_yaw_angle (float, optional): Maximum constraint on 
                yaw. Defaults to None.
            x0 (iterable, optional): The initial yaw conditions. 
                Defaults to None. Initializes to the current turbine 
                yaw settings.
            bnds (iterable, optional): Bounds for the optimization 
                variables (pairs of min/max values for each variable). 
                Defaults to None. Initializes to [(0.0, 25.0)].
            opt_method (str, optional): The optimization method for 
                scipy.optimize.minize to use. Defaults to None. 
                Initializes to 'SLSQP'.
            opt_options (dictionary, optional): Optimization options for 
                scipy.optimize.minize to use. Defaults to None. 
                Initializes to {'maxiter': 100, 'disp': False,
                'iprint': 1, 'ftol': 1e-7, 'eps': 0.01}.
            include_unc (bool): If True, uncertainty in wind direction 
                and/or yaw position is included when determining wind farm power. 
                Uncertainty is included by computing the mean wind farm power for 
                a distribution of wind direction and yaw position deviations from 
                the original wind direction and yaw angles. Defaults to None.
            unc_pmfs (dictionary, optional): A dictionary containing optional 
                probability mass functions describing the distribution of wind 
                direction and yaw position deviations when wind direction and/or 
                yaw position uncertainty is included in the power calculations. 
                Contains the following key-value pairs:  

                -   **wd_unc**: A numpy array containing wind direction deviations 
                    from the original wind direction. 
                -   **wd_unc_pmf**: A numpy array containing the probability of 
                    each wind direction deviation in **wd_unc** occuring. 
                -   **yaw_unc**: A numpy array containing yaw angle deviations 
                    from the original yaw angles. 
                -   **yaw_unc_pmf**: A numpy array containing the probability of 
                    each yaw angle deviation in **yaw_unc** occuring.

                Defaults to None. If the object's **include_unc** parameter is True 
                and **unc_pmfs** has not been initialized, initializes using 
                normally-distributed, zero-mean PMFs based on the values in 
                **unc_options**.
            unc_options (disctionary, optional): A dictionary containing values used 
                to create normally-distributed, zero-mean probability mass functions 
                describing the distribution of wind direction and yaw position 
                deviations when wind direction and/or yaw position uncertainty is 
                included. This argument is only used when **unc_pmfs** is None and 
                contains the following key-value pairs:

                -   **std_wd**: A float containing the standard deviation of the wind 
                        direction deviations from the original wind direction.
                -   **std_yaw**: A float containing the standard deviation of the yaw 
                        angle deviations from the original yaw angles.
                -   **pmf_res**: A float containing the resolution in degrees of the 
                        wind direction and yaw angle PMFs.
                -   **pdf_cutoff**: A float containing the cumulative distribution 
                    function value at which the tails of the PMFs are truncated. 

                Defaults to None. Initializes to {'std_wd': 4.95, 'std_yaw': 1.75, 
                'pmf_res': 1.0, 'pdf_cutoff': 0.995}.
        """
        if minimum_yaw_angle is not None:
            self.minimum_yaw_angle = minimum_yaw_angle
        if maximum_yaw_angle is not None:
            self.maximum_yaw_angle = maximum_yaw_angle
        if x0 is not None:
            self.x0 = x0
        else:
            self.x0 = [turbine.yaw_angle for turbine in \
                       self.fi.floris.farm.turbine_map.turbines]
        if bnds is not None:
            self.bnds = bnds
        else:
            self._set_opt_bounds(self.minimum_yaw_angle, 
                                 self.maximum_yaw_angle)
        if opt_method is not None:
            self.opt_method = opt_method
        if opt_options is not None:
            self.opt_options = opt_options
        if include_unc is not None:
            self.include_unc = include_unc
        if unc_pmfs is not None:
            self.unc_pmfs = unc_pmfs
        if unc_options is not None:
            self.unc_options = unc_options

        if self.include_unc & (self.unc_pmfs is None):
            if self.unc_options is None:
                self.unc_options = {'std_wd': 4.95, 'std_yaw': 1.75, \
                            'pmf_res': 1.0, 'pdf_cutoff': 0.995}

            # create normally distributed wd and yaw uncertaitny pmfs
            if self.unc_options['std_wd'] > 0:
                wd_bnd = int(np.ceil(norm.ppf(self.unc_options['pdf_cutoff'], \
                                scale=self.unc_options['std_wd'])/self.unc_options['pmf_res']))
                wd_unc = np.linspace(-1*wd_bnd*self.unc_options['pmf_res'],wd_bnd*self.unc_options['pmf_res'],2*wd_bnd+1)
                wd_unc_pmf = norm.pdf(wd_unc,scale=self.unc_options['std_wd'])
                wd_unc_pmf = wd_unc_pmf / np.sum(wd_unc_pmf) # normalize so sum = 1.0
            else:
                wd_unc = np.zeros(1)
                wd_unc_pmf = np.ones(1)

            if self.unc_options['std_yaw'] > 0:
                yaw_bnd = int(np.ceil(norm.ppf(self.unc_options['pdf_cutoff'], \
                                scale=self.unc_options['std_yaw'])/self.unc_options['pmf_res']))
                yaw_unc = np.linspace(-1*yaw_bnd*self.unc_options['pmf_res'],yaw_bnd*self.unc_options['pmf_res'],2*yaw_bnd+1)
                yaw_unc_pmf = norm.pdf(yaw_unc,scale=self.unc_options['std_yaw'])
                yaw_unc_pmf = yaw_unc_pmf / np.sum(yaw_unc_pmf) # normalize so sum = 1.0
            else:
                yaw_unc = np.zeros(1)
                yaw_unc_pmf = np.ones(1)

            self.unc_pmfs = {'wd_unc': wd_unc, 'wd_unc_pmf': wd_unc_pmf, \
                        'yaw_unc': yaw_unc, 'yaw_unc_pmf': yaw_unc_pmf}

    # Properties

    @property
    def minimum_yaw_angle(self):
        """
        This property gets or sets the minimum yaw angle for the 
        optimization and updates the bounds accordingly.
        
        Args:
            value (float): The minimum yaw angle (deg).

        Returns:
            minimum_yaw_angle (float): The minimum yaw angle (deg).
        """
        return self._minimum_yaw_angle

    @minimum_yaw_angle.setter
    def minimum_yaw_angle(self, value):
        if not hasattr(self, 'maximum_yaw_angle'):
            self._set_opt_bounds(value, 25.0)
        else:
            self._set_opt_bounds(value, self.maximum_yaw_angle)
        self._minimum_yaw_angle = value

    @property
    def maximum_yaw_angle(self):
        """
        This property gets or sets the maximum yaw angle for the 
        optimization and updates the bounds accordingly.
        
        Args:
            value (float): The maximum yaw angle (deg).

        Returns:
            minimum_yaw_angle (float): The maximum yaw angle (deg).
        """
        return self._maximum_yaw_angle

    @maximum_yaw_angle.setter
    def maximum_yaw_angle(self, value):
        if not hasattr(self, 'minimum_yaw_angle'):
            self._set_opt_bounds(0.0, value)
        else:
            self._set_opt_bounds(self.minimum_yaw_angle, value)
        self._maximum_yaw_angle = value

    @property
    def x0(self):
        """
        This property gets or sets the initial yaw angles for the 
        optimization.
        
        Args:
            value (float): The initial yaw angles (deg).

        Returns:
            x0 (float): The initial yaw angles (deg).
        """
        return self._x0

    @x0.setter
    def x0(self, value):
        self._x0 = value

class YawOptimizationWindRose(Optimization):
    """
    Sub class of the :py:class`floris.tools.optimization.Optimization`
    object class that performs yaw optimization for multiple wind speed,
    wind direction combinations.

    Args:
        fi (:py:class:`floris.tools.floris_utilities.FlorisInterface`): 
            Interface from FLORIS to the tools package.
        minimum_yaw_angle (float, optional): Minimum constraint on 
            yaw. Defaults to None.
        maximum_yaw_angle (float, optional): Maximum constraint on 
            yaw. Defaults to None.
        minimum_ws (float, optional): Minimum wind speed at which optimization is performed. 
            Assume zero power generated below this value. Defaults to 3 m/s.
        maximum_ws (float, optional): Maximum wind speed at which optimization is performed. 
            Assume optimal yaw offsets are zero above this wind speed. Defaults to 25 m/s.
        wd (np.array) : The wind directions for the AEP optimization.
        ws (np.array): The wind speeds for the AEP optimization.
        x0 (iterable, optional): The initial yaw conditions. 
            Defaults to None. Initializes to the current turbine 
            yaw settings.
        bnds (iterable, optional): Bounds for the optimization 
            variables (pairs of min/max values for each variable). 
            Defaults to None. Initializes to [(0.0, 25.0)].
        opt_method (str, optional): The optimization method for 
            scipy.optimize.minize to use. Defaults to None. 
            Initializes to 'SLSQP'.
        opt_options (dictionary, optional): Optimization options for 
            scipy.optimize.minize to use. Defaults to None. 
            Initializes to {'maxiter': 100, 'disp': False,
            'iprint': 1, 'ftol': 1e-7, 'eps': 0.01}.
        include_unc (bool): If True, uncertainty in wind direction 
            and/or yaw position is included when determining wind farm power. 
            Uncertainty is included by computing the mean wind farm power for 
            a distribution of wind direction and yaw position deviations from 
            the original wind direction and yaw angles. Defaults to False.
        unc_pmfs (dictionary, optional): A dictionary containing optional 
            probability mass functions describing the distribution of wind 
            direction and yaw position deviations when wind direction and/or 
            yaw position uncertainty is included in the power calculations. 
            Contains the following key-value pairs:  

            -   **wd_unc**: A numpy array containing wind direction deviations 
                from the original wind direction. 
            -   **wd_unc_pmf**: A numpy array containing the probability of 
                each wind direction deviation in **wd_unc** occuring. 
            -   **yaw_unc**: A numpy array containing yaw angle deviations 
                from the original yaw angles. 
            -   **yaw_unc_pmf**: A numpy array containing the probability of 
                each yaw angle deviation in **yaw_unc** occuring.

            Defaults to None, in which case default PMFs are calculated using 
            values provided in **unc_options**.
        unc_options (disctionary, optional): A dictionary containing values used 
            to create normally-distributed, zero-mean probability mass functions 
            describing the distribution of wind direction and yaw position 
            deviations when wind direction and/or yaw position uncertainty is 
            included. This argument is only used when **unc_pmfs** is None and 
            contains the following key-value pairs:

            -   **std_wd**: A float containing the standard deviation of the wind 
                    direction deviations from the original wind direction.
            -   **std_yaw**: A float containing the standard deviation of the yaw 
                    angle deviations from the original yaw angles.
            -   **pmf_res**: A float containing the resolution in degrees of the 
                    wind direction and yaw angle PMFs.
            -   **pdf_cutoff**: A float containing the cumulative distribution 
                function value at which the tails of the PMFs are truncated. 

            Defaults to None. Initializes to {'std_wd': 4.95, 'std_yaw': 1.75, 
            'pmf_res': 1.0, 'pdf_cutoff': 0.995}.

    Returns:
        YawOptimizationWindRose: An instantiated YawOptimizationWindRose object.
    """

    def __init__(self, fi, wd,
                           ws,
                           minimum_yaw_angle=0.0,
                           maximum_yaw_angle=25.0,
                           minimum_ws=3.0,
                           maximum_ws=25.0,
                           x0=None,
                           bnds=None,
                           opt_method='SLSQP',
                           opt_options=None,
                           include_unc=False,
                           unc_pmfs=None,
                           unc_options=None):
        """
        Instantiate YawOptimizationWindRose object and parameter values.
        """
        super().__init__(fi)
        
        if opt_options is None:
            self.opt_options = {'maxiter': 100, 'disp': False, \
                        'iprint': 1, 'ftol': 1e-7, 'eps': 0.01}

        self.unc_pmfs = unc_pmfs

        if unc_options is None:
            self.unc_options = {'std_wd': 4.95, 'std_yaw': 1.75, \
                        'pmf_res': 1.0, 'pdf_cutoff': 0.995}

        self.reinitialize_opt_wind_rose(
            wd=wd,
            ws=ws,
            minimum_yaw_angle=minimum_yaw_angle,
            maximum_yaw_angle=maximum_yaw_angle,
            minimum_ws=minimum_ws,
            maximum_ws=maximum_ws,
            x0=x0,
            bnds=bnds,
            opt_method=opt_method,
            opt_options=opt_options,
            include_unc=include_unc,
            unc_pmfs=unc_pmfs,
            unc_options=unc_options
        )

    # Private methods

    def _get_power_for_yaw_angle_opt(self, yaw_angles):
        """
        Assign yaw angles to turbines, calculate wake, report power

        Args:
            yaw_angles (np.array): yaw to apply to each turbine

        Returns:
            power (float): wind plant power. #TODO negative? in kW?
        """

        power = -1 * self.fi.get_farm_power_for_yaw_angle(yaw_angles, \
            include_unc=self.include_unc, unc_pmfs=self.unc_pmfs, \
            unc_options=self.unc_options)

        return power / (10**3)

    def _set_opt_bounds(self, minimum_yaw_angle, maximum_yaw_angle):
        """
        Sets minimum and maximum yaw angle bounds for optimization.
        """

        self.bnds = [(minimum_yaw_angle, maximum_yaw_angle) for _ in \
                     range(self.nturbs)]

    def _optimize(self):
        """
        Find optimum setting of turbine yaw angles for power production
        given fixed atmospheric conditins (wind speed, direction, etc.).

        Returns:
            opt_yaw_angles (np.array): optimal yaw angles of each turbine.
        """
        self.residual_plant = minimize(self._get_power_for_yaw_angle_opt,
                                self.x0,
                                method=self.opt_method,
                                bounds=self.bnds,
                                options=self.opt_options)

        opt_yaw_angles = self.residual_plant.x

        return opt_yaw_angles

    # Public methods

    def reinitialize_opt_wind_rose(self,
            wd=None,
            ws=None,
            minimum_yaw_angle=None,
            maximum_yaw_angle=None,
            minimum_ws=None,
            maximum_ws=None,
            x0=None,
            bnds=None,
            opt_method=None,
            opt_options=None,
            include_unc=None,
            unc_pmfs=None,
            unc_options=None):
        """
        Reintializes parameter values for the optimization.
        
        This method reinitializes the optimization parameters and 
        bounds to the supplied values or uses what is currently stored.
        
        Args:
            fi (:py:class:`floris.tools.floris_utilities.FlorisInterface`): 
                Interface from FLORIS to the tools package.
            minimum_yaw_angle (float, optional): Minimum constraint on 
                yaw. Defaults to None.
            maximum_yaw_angle (float, optional): Maximum constraint on 
                yaw. Defaults to None.
            minimum_ws (float, optional): Minimum wind speed at which optimization is performed. 
                Assume zero power generated below this value. Defaults to None.
            maximum_ws (float, optional): Maximum wind speed at which optimization is performed. 
                Assume optimal yaw offsets are zero above this wind speed. Defaults to None.
            wd (np.array) : The wind directions for the AEP optimization.
            ws (np.array): The wind speeds for the AEP optimization.
            x0 (iterable, optional): The initial yaw conditions. 
                Defaults to None. Initializes to the current turbine 
                yaw settings.
            bnds (iterable, optional): Bounds for the optimization 
                variables (pairs of min/max values for each variable). 
                Defaults to None. Initializes to [(0.0, 25.0)].
            opt_method (str, optional): The optimization method for 
                scipy.optimize.minize to use. Defaults to None. 
                Initializes to 'SLSQP'.
            opt_options (dictionary, optional): Optimization options for 
                scipy.optimize.minize to use. Defaults to None. 
                Initializes to {'maxiter': 100, 'disp': False,
                'iprint': 1, 'ftol': 1e-7, 'eps': 0.01}.
            include_unc (bool): If True, uncertainty in wind direction 
                and/or yaw position is included when determining wind farm power. 
                Uncertainty is included by computing the mean wind farm power for 
                a distribution of wind direction and yaw position deviations from 
                the original wind direction and yaw angles. Defaults to None.
            unc_pmfs (dictionary, optional): A dictionary containing optional 
                probability mass functions describing the distribution of wind 
                direction and yaw position deviations when wind direction and/or 
                yaw position uncertainty is included in the power calculations. 
                Contains the following key-value pairs:  

                -   **wd_unc**: A numpy array containing wind direction deviations 
                    from the original wind direction. 
                -   **wd_unc_pmf**: A numpy array containing the probability of 
                    each wind direction deviation in **wd_unc** occuring. 
                -   **yaw_unc**: A numpy array containing yaw angle deviations 
                    from the original yaw angles. 
                -   **yaw_unc_pmf**: A numpy array containing the probability of 
                    each yaw angle deviation in **yaw_unc** occuring.

                Defaults to None. If the object's **include_unc** parameter is True 
                and **unc_pmfs** has not been initialized, initializes using 
                normally-distributed, zero-mean PMFs based on the values in 
                **unc_options**.
            unc_options (disctionary, optional): A dictionary containing values used 
                to create normally-distributed, zero-mean probability mass functions 
                describing the distribution of wind direction and yaw position 
                deviations when wind direction and/or yaw position uncertainty is 
                included. This argument is only used when **unc_pmfs** is None and 
                contains the following key-value pairs:

                -   **std_wd**: A float containing the standard deviation of the wind 
                        direction deviations from the original wind direction.
                -   **std_yaw**: A float containing the standard deviation of the yaw 
                        angle deviations from the original yaw angles.
                -   **pmf_res**: A float containing the resolution in degrees of the 
                        wind direction and yaw angle PMFs.
                -   **pdf_cutoff**: A float containing the cumulative distribution 
                    function value at which the tails of the PMFs are truncated. 

                Defaults to None. Initializes to {'std_wd': 4.95, 'std_yaw': 1.75, 
                'pmf_res': 1.0, 'pdf_cutoff': 0.995}.
        """

        if wd is not None:
            self.wd = wd
        if ws is not None:
            self.ws = ws
        if minimum_ws is not None:
            self.minimum_ws = minimum_ws
        if maximum_ws is not None:
            self.maximum_ws = maximum_ws
        if minimum_yaw_angle is not None:
            self.minimum_yaw_angle = minimum_yaw_angle
        if maximum_yaw_angle is not None:
            self.maximum_yaw_angle = maximum_yaw_angle
        if opt_method is not None:
            self.opt_method = opt_method
        if opt_options is not None:
            self.opt_options = opt_options
        if x0 is not None:
            self.x0 = x0
        else:
            self.x0 = [turbine.yaw_angle for turbine in \
                       self.fi.floris.farm.turbine_map.turbines]
        if bnds is not None:
            self.bnds = bnds
        else:
            self._set_opt_bounds(self.minimum_yaw_angle, 
                                 self.maximum_yaw_angle)
        if include_unc is not None:
            self.include_unc = include_unc
        if unc_pmfs is not None:
            self.unc_pmfs = unc_pmfs
        if unc_options is not None:
            self.unc_options = unc_options

        if self.include_unc & (self.unc_pmfs is None):
            if self.unc_options is None:
                self.unc_options = {'std_wd': 4.95, 'std_yaw': 1.75, \
                            'pmf_res': 1.0, 'pdf_cutoff': 0.995}

            # create normally distributed wd and yaw uncertaitny pmfs
            if self.unc_options['std_wd'] > 0:
                wd_bnd = int(np.ceil(norm.ppf(self.unc_options['pdf_cutoff'], \
                                scale=self.unc_options['std_wd'])/self.unc_options['pmf_res']))
                wd_unc = np.linspace(-1*wd_bnd*self.unc_options['pmf_res'],wd_bnd*self.unc_options['pmf_res'],2*wd_bnd+1)
                wd_unc_pmf = norm.pdf(wd_unc,scale=self.unc_options['std_wd'])
                wd_unc_pmf = wd_unc_pmf / np.sum(wd_unc_pmf) # normalize so sum = 1.0
            else:
                wd_unc = np.zeros(1)
                wd_unc_pmf = np.ones(1)

            if self.unc_options['std_yaw'] > 0:
                yaw_bnd = int(np.ceil(norm.ppf(self.unc_options['pdf_cutoff'], \
                                scale=self.unc_options['std_yaw'])/self.unc_options['pmf_res']))
                yaw_unc = np.linspace(-1*yaw_bnd*self.unc_options['pmf_res'],yaw_bnd*self.unc_options['pmf_res'],2*yaw_bnd+1)
                yaw_unc_pmf = norm.pdf(yaw_unc,scale=self.unc_options['std_yaw'])
                yaw_unc_pmf = yaw_unc_pmf / np.sum(yaw_unc_pmf) # normalize so sum = 1.0
            else:
                yaw_unc = np.zeros(1)
                yaw_unc_pmf = np.ones(1)

            self.unc_pmfs = {'wd_unc': wd_unc, 'wd_unc_pmf': wd_unc_pmf, \
                        'yaw_unc': yaw_unc, 'yaw_unc_pmf': yaw_unc_pmf}

    def calc_baseline_power(self):
        """
        For a series of (wind speed, direction) pairs, finds the baseline power produced by the wind farm and 
        the ideal power without wake losses. 

        Returns:
            - **df_base** (*Pandas DataFrame*) - DataFrame with the same number of rows as the length of the wd 
              and ws arrays, containing the following columns:

                - **ws** (*float*) - The wind speed value for the row.
                - **wd** (*float*) - The wind direction value for the row.
                - **power_baseline** (*float*) - The total power produced by the wind farm with baseline 
                  yaw control (W).
                - **power_no_wake** (*float*) - The ideal total power produced by the wind farm without 
                  wake losses (W).
                - **turbine_power_baseline** (*list* of *float* values) - A list containing the baseline power 
                  without wake steering for each wind turbine (W).
                - **turbine_power_no_wake** (*list* of *float* values) - A list containing the ideal power 
                  without wake losses for each wind turbine (W).
        """
        #print('=====================================================')
        #print('Calculating baseline power...')
        #print('Number of wind speed, wind direction pairs to calculate = ', len(self.wd))
        #print('=====================================================')

        df_base = pd.DataFrame()

        for i in range(len(self.wd)):
            #print('Computing wind speed, wind direction pair '+str(i)+' out of '+str(len(self.wd)) \
            #    +': wind speed = '+str(self.ws[i])+' m/s, wind direction = '+str(self.wd[i])+' deg.')

            # Find baseline power in FLORIS

            if self.ws[i] >= self.minimum_ws:
                self.fi.reinitialize_flow_field(
                    wind_direction=self.wd[i], wind_speed=self.ws[i])
                
                # calculate baseline power
                self.fi.calculate_wake(yaw_angles=0.0)
                power_base = self.fi.get_turbine_power(include_unc=self.include_unc, \
                    unc_pmfs=self.unc_pmfs, unc_options=self.unc_options)

                # calculate power for no wake case
                self.fi.calculate_wake(no_wake=True)
                power_no_wake = self.fi.get_turbine_power(include_unc=self.include_unc, \
                    unc_pmfs=self.unc_pmfs, unc_options=self.unc_options, no_wake=True)
            else:
                power_base = self.nturbs*[0.0]
                power_no_wake = self.nturbs*[0.0]

            # add variables to dataframe
            df_base = df_base.append(pd.DataFrame({'ws':[self.ws[i]],'wd':[self.wd[i]], \
                'power_baseline':[np.sum(power_base)],'turbine_power_baseline':[power_base], \
                'power_no_wake':[np.sum(power_no_wake)],'turbine_power_no_wake':[power_no_wake]}))

        df_base.reset_index(drop=True,inplace=True)

        return df_base

    def optimize(self):
        """
        For a series of (wind speed, direction) pairs, finds the power resulting from optimal wake steering. 

        Returns:
            - **df_opt** (*Pandas DataFrame*) - DataFrame with the same number of rows as the length of the wd 
              and ws arrays, containing the following columns:

                - **ws** (*float*) - The wind speed value for the row.
                - **wd** (*float*) - The wind direction value for the row.
                - **power_opt** (*float*) - The total power produced by the wind farm with optimal 
                  yaw offsets (W).
                - **turbine_power_opt** (*list* of *float* values) - A list containing the power produced by
                  each wind turbine with optimal yaw offsets (W).
                - **yaw_angles** (*list* of *float* values) - A list containing the optimal yaw offsets for maximizing 
                  total wind farm power for each wind turbine (deg).
        """
        print('=====================================================')
        print('Optimizing wake redirection control...')
        print('Number of wind speed, wind direction pairs to optimize = ', len(self.wd))
        print('Number of yaw angles to optimize = ', len(self.x0))
        print('=====================================================')

        df_opt = pd.DataFrame()

        for i in range(len(self.wd)):
            print('Computing wind speed, wind direction pair '+str(i)+' out of '+str(len(self.wd)) \
                +': wind speed = '+str(self.ws[i])+' m/s, wind direction = '+str(self.wd[i])+' deg.')

            # Optimizing wake redirection control

            if (self.ws[i] >= self.minimum_ws) & (self.ws[i] <= self.maximum_ws):
                self.fi.reinitialize_flow_field(
                    wind_direction=self.wd[i], wind_speed=self.ws[i])

                opt_yaw_angles = self._optimize()

                if np.sum(opt_yaw_angles) == 0:
                    print('No change in controls suggested for this inflow \
                        condition...')

                # optimized power
                self.fi.calculate_wake(yaw_angles=opt_yaw_angles)
                power_opt = self.fi.get_turbine_power(include_unc=self.include_unc, \
                    unc_pmfs=self.unc_pmfs, unc_options=self.unc_options)
            elif self.ws[i] >= self.minimum_ws:
                print('No change in controls suggested for this inflow \
                        condition...')
                self.fi.reinitialize_flow_field(
                    wind_direction=self.wd[i], wind_speed=self.ws[i])
                self.fi.calculate_wake(yaw_angles=0.0)
                opt_yaw_angles = self.nturbs*[0.0]
                power_opt = self.fi.get_turbine_power(include_unc=self.include_unc, \
                    unc_pmfs=self.unc_pmfs, unc_options=self.unc_options)
            else:
                print('No change in controls suggested for this inflow \
                        condition...')
                opt_yaw_angles = self.nturbs*[0.0]
                power_opt = self.nturbs*[0.0]

            # add variables to dataframe
            df_opt = df_opt.append(pd.DataFrame({'ws':[self.ws[i]],'wd':[self.wd[i]], \
                'power_opt':[np.sum(power_opt)],'turbine_power_opt':[power_opt],'yaw_angles':[opt_yaw_angles]}))

        df_opt.reset_index(drop=True,inplace=True)

        return df_opt


class YawOptimizationWindRoseParallel(YawOptimizationWindRose):
    """
    Sub class of the :py:class`floris.tools.optimization.YawOptimizationWindRose`
    object class that performs parallel yaw optimization for multiple wind speed,
    wind direction combinations using the MPIPoolExecutor method of the 
    mpi4py.futures module.

    Args:
        fi (:py:class:`floris.tools.floris_utilities.FlorisInterface`): 
            Interface from FLORIS to the tools package.
        minimum_yaw_angle (float, optional): Minimum constraint on 
            yaw. Defaults to None.
        maximum_yaw_angle (float, optional): Maximum constraint on 
            yaw. Defaults to None.
        minimum_ws (float, optional): Minimum wind speed at which optimization is performed. 
            Assume zero power generated below this value. Defaults to 3 m/s.
        maximum_ws (float, optional): Maximum wind speed at which optimization is performed. 
            Assume optimal yaw offsets are zero above this wind speed. Defaults to 25 m/s.
        wd (np.array): The wind directions for the AEP optimization.
        ws (np.array): The wind speeds for the AEP optimization.
        x0 (iterable, optional): The initial yaw conditions. 
            Defaults to None. Initializes to the current turbine 
            yaw settings.
        bnds (iterable, optional): Bounds for the optimization 
            variables (pairs of min/max values for each variable). 
            Defaults to None. Initializes to [(0.0, 25.0)].
        opt_method (str, optional): The optimization method for 
            scipy.optimize.minize to use. Defaults to None. 
            Initializes to 'SLSQP'.
        opt_options (dictionary, optional): Optimization options for 
            scipy.optimize.minize to use. Defaults to None. 
            Initializes to {'maxiter': 100, 'disp': False,
            'iprint': 1, 'ftol': 1e-7, 'eps': 0.01}.
        include_unc (bool): If True, uncertainty in wind direction 
            and/or yaw position is included when determining wind farm power. 
            Uncertainty is included by computing the mean wind farm power for 
            a distribution of wind direction and yaw position deviations from 
            the original wind direction and yaw angles. Defaults to False.
        unc_pmfs (dictionary, optional): A dictionary containing optional 
            probability mass functions describing the distribution of wind 
            direction and yaw position deviations when wind direction and/or 
            yaw position uncertainty is included in the power calculations. 
            Contains the following key-value pairs:  

            -   **wd_unc**: A numpy array containing wind direction deviations 
                from the original wind direction. 
            -   **wd_unc_pmf**: A numpy array containing the probability of 
                each wind direction deviation in **wd_unc** occuring. 
            -   **yaw_unc**: A numpy array containing yaw angle deviations 
                from the original yaw angles. 
            -   **yaw_unc_pmf**: A numpy array containing the probability of 
                each yaw angle deviation in **yaw_unc** occuring.

            Defaults to None, in which case default PMFs are calculated using 
            values provided in **unc_options**.
        unc_options (disctionary, optional): A dictionary containing values used 
            to create normally-distributed, zero-mean probability mass functions 
            describing the distribution of wind direction and yaw position 
            deviations when wind direction and/or yaw position uncertainty is 
            included. This argument is only used when **unc_pmfs** is None and 
            contains the following key-value pairs:

            -   **std_wd**: A float containing the standard deviation of the wind 
                    direction deviations from the original wind direction.
            -   **std_yaw**: A float containing the standard deviation of the yaw 
                    angle deviations from the original yaw angles.
            -   **pmf_res**: A float containing the resolution in degrees of the 
                    wind direction and yaw angle PMFs.
            -   **pdf_cutoff**: A float containing the cumulative distribution 
                function value at which the tails of the PMFs are truncated. 

            Defaults to None. Initializes to {'std_wd': 4.95, 'std_yaw': 1.75, 
            'pmf_res': 1.0, 'pdf_cutoff': 0.995}.

    Returns:
        YawOptimizationWindRoseParallel: An instantiated YawOptimizationWindRoseParallel object.
    """

    # Private methods

    def _calc_baseline_power_one_case(self,ws,wd):
        """
        For a single (wind speed, direction) pair, finds the baseline power produced by the wind farm 
        and the ideal power without wake losses.

        Args:
            ws (float): The wind speed used in floris for the yaw optimization.
            wd (float): The wind direction used in floris for the yaw optimization.

        Returns:
            - **df_base** (*Pandas DataFrame*) - DataFrame with a single row, containing the following columns:

                - **ws** (*float*) - The wind speed value for the row.
                - **wd** (*float*) - The wind direction value for the row.
                - **power_baseline** (*float*) - The total power produced by the wind farm with baseline 
                  yaw control (W).
                - **power_no_wake** (*float*) - The ideal total power produced by the wind farm without 
                  wake losses (W).
                - **turbine_power_baseline** (*list* of *float* values) - A list containing the baseline power 
                  without wake steering for each wind turbine (W).
                - **turbine_power_no_wake** (*list* of *float* values) - A list containing the ideal power 
                  without wake losses for each wind turbine (W).
        """

        print('Computing wind speed = '+str(ws)+' m/s, wind direction = '+str(wd)+' deg.')

        # Find baseline power in FLORIS

        if ws >= self.minimum_ws:
            self.fi.reinitialize_flow_field(wind_direction=wd, wind_speed=ws)
            # calculate baseline power
            self.fi.calculate_wake(yaw_angles=0.0)
            power_base = self.fi.get_turbine_power(include_unc=self.include_unc, \
                    unc_pmfs=self.unc_pmfs, unc_options=self.unc_options)

            # calculate power for no wake case
            self.fi.calculate_wake(no_wake=True)
            power_no_wake = self.fi.get_turbine_power(include_unc=self.include_unc, \
                    unc_pmfs=self.unc_pmfs, unc_options=self.unc_options, no_wake=True)
        else:
            power_base = self.nturbs*[0.0]
            power_no_wake = self.nturbs*[0.0]

        # add variables to dataframe
        df_base = pd.DataFrame({'ws':[ws],'wd':[wd], \
            'power_baseline':[np.sum(power_base)],'turbine_power_baseline':[power_base], \
            'power_no_wake':[np.sum(power_no_wake)],'turbine_power_no_wake':[power_no_wake]})

        return df_base

    def _optimize_one_case(self,ws,wd):
        """
        For a single (wind speed, direction) pair, finds the power resulting from optimal wake steering.

        Args:
            ws (float): The wind speed used in floris for the yaw optimization.
            wd (float): The wind direction used in floris for the yaw optimization.

        Returns:
            - **df_opt** (*Pandas DataFrame*) - DataFrame with a single row, containing the following columns:

                - **ws** (*float*) - The wind speed value for the row.
                - **wd** (*float*) - The wind direction value for the row.
                - **power_opt** (*float*) - The total power produced by the wind farm with optimal 
                  yaw offsets (W).
                - **turbine_power_opt** (*list* of *float* values) - A list containing the power produced by
                  each wind turbine with optimal yaw offsets (W).
                - **yaw_angles** (*list* of *float* values) - A list containing the optimal yaw offsets for maximizing 
                  total wind farm power for each wind turbine (deg).
        """

        print('Computing wind speed = '+str(ws)+' m/s, wind direction = '+str(wd)+' deg.')

        # Optimizing wake redirection control

        if (ws >= self.minimum_ws) & (ws <= self.maximum_ws):
            self.fi.reinitialize_flow_field(
                wind_direction=wd, wind_speed=ws)

            opt_yaw_angles = self._optimize()

            if np.sum(opt_yaw_angles) == 0:
                print('No change in controls suggested for this inflow \
                    condition...')

            # optimized power
            self.fi.calculate_wake(yaw_angles=opt_yaw_angles)
            power_opt = self.fi.get_turbine_power(include_unc=self.include_unc, \
                    unc_pmfs=self.unc_pmfs, unc_options=self.unc_options)
        elif ws >= self.minimum_ws:
            print('No change in controls suggested for this inflow \
                    condition...')
            self.fi.reinitialize_flow_field(
                wind_direction=wd, wind_speed=ws)
            self.fi.calculate_wake(yaw_angles=0.0)
            opt_yaw_angles = self.nturbs*[0.0]
            power_opt = self.fi.get_turbine_power(include_unc=self.include_unc, \
                    unc_pmfs=self.unc_pmfs, unc_options=self.unc_options)
        else:
            print('No change in controls suggested for this inflow \
                    condition...')
            opt_yaw_angles = self.nturbs*[0.0]
            power_opt = self.nturbs*[0.0]

        # add variables to dataframe
        df_opt = pd.DataFrame({'ws':[ws],'wd':[wd], \
            'power_opt':[np.sum(power_opt)],'turbine_power_opt':[power_opt],'yaw_angles':[opt_yaw_angles]})

        return df_opt

    # Public methods    

    def calc_baseline_power(self):
        """
        For a series of (wind speed, direction) pairs, finds the baseline power produced by the wind farm 
        and the ideal power without wake losses. The optimization for different wind speed, wind direction 
        combinations is parallelized using the mpi4py.futures module.

        Returns:
            - **df_base** (*Pandas DataFrame*) - DataFrame with the same number of rows as the length of the wd 
              and ws arrays, containing the following columns:

                - **ws** (*float*) - The wind speed value for the row.
                - **wd** (*float*) - The wind direction value for the row.
                - **power_baseline** (*float*) - The total power produced by the wind farm with baseline 
                  yaw control (W).
                - **power_no_wake** (*float*) - The ideal total power produced by the wind farm without 
                  wake losses (W).
                - **turbine_power_baseline** (*list* of *float* values) - A list containing the baseline power 
                  without wake steering for each wind turbine (W).
                - **turbine_power_no_wake** (*list* of *float* values) - A list containing the ideal power 
                  without wake losses for each wind turbine (W).
        """

        print('=====================================================')
        print('Calculating baseline power in parallel...')
        print('Number of wind speed, wind direction pairs to calculate = ', len(self.wd))
        print('=====================================================')

        df_base = pd.DataFrame()

        with MPIPoolExecutor() as executor: 
            for df_base_one in executor.map(self._calc_baseline_power_one_case,self.ws.values,self.wd.values):
            
                # add variables to dataframe
                df_base = df_base.append(df_base_one)
        
        df_base.reset_index(drop=True,inplace=True)

        return df_base

    def optimize(self):
        """
        For a series of (wind speed, direction) pairs, finds the power resulting from optimal wake steering. 
        The optimization for different wind speed, wind direction combinations is parallelized using the 
        mpi4py.futures module.

        Returns:
            - **df_opt** (*Pandas DataFrame*) - DataFrame with the same number of rows as the length of the wd 
              and ws arrays, containing the following columns:

                - **ws** (*float*) - The wind speed value for the row.
                - **wd** (*float*) - The wind direction value for the row.
                - **power_opt** (*float*) - The total power produced by the wind farm with optimal 
                  yaw offsets (W).
                - **turbine_power_opt** (*list* of *float* values) - A list containing the power produced by
                  each wind turbine with optimal yaw offsets (W).
                - **yaw_angles** (*list* of *float* values) - A list containing the optimal yaw offsets for maximizing 
                  total wind farm power for each wind turbine (deg).
        """

        print('=====================================================')
        print('Optimizing wake redirection control in parallel...')
        print('Number of wind speed, wind direction pairs to optimize = ', len(self.wd))
        print('Number of yaw angles to optimize = ', len(self.x0))
        print('=====================================================')

        df_opt = pd.DataFrame()

        with MPIPoolExecutor() as executor: 
            for df_opt_one in executor.map(self._optimize_one_case,self.ws.values,self.wd.values):
            
                # add variables to dataframe
                df_opt = df_opt.append(df_opt_one)
        
        df_opt.reset_index(drop=True,inplace=True)

        return df_opt


class LayoutOptimization(Optimization):
    """
    Sub class of the :py:class`floris.tools.optimization.Optimization`
    object class that performs layout optimization.
    """

    def __init__(self, fi, boundaries,
                           wd,
                           ws,
                           freq,
                           AEP_initial,
                           x0=None,
                           bnds=None,
                           min_dist=None,
                           opt_method='SLSQP',
                           opt_options=None):
        """
        Instantiate LayoutOptimization object and parameter values.
        
        Args:
            fi (:py:class:`floris.tools.floris_utilities.FlorisInterface`): 
                Interface from FLORIS to the tools package.
            boundaries (iterable): A list of pairs of floats that 
                represent the boundary's vertices. Defaults to None.
            wd (np.array): An array of wind directions. 
                Defaults to None.
            ws (np.array): An array of wind speeds. Defaults 
                to None.
            freq (np.array): An array of wind direction 
                frequency values. Defaults to None.
            AEP_initial (float): Initial Annual Energy 
                Production used in normalizing optimization. Defaults 
                to None. Initializes to the AEP of the current FLORIS 
                object.
            x0 (iterable, optional): The initial turbine locations, 
                ordered by x-coordinate and then y-coordiante 
                (ie. [x1, x2, ..., xn, y1, y2, ..., yn]). Defaults to 
                None. Initializes to the current turbine locations.
            bnds (iterable, optional): Bounds for the optimization 
                variables (pairs of min/max values for each variable). 
                Defaults to None. Initializes to the min. and max. 
                values of the boundaries iterable.
            min_dist (float, optional): The minimum distance to be 
                maitained between turbines during the optimization. 
                Defaults to None. Initializes to 2 rotor diamters.
            opt_method (str, optional): The optimization method for 
                scipy.optimize.minize to use. Defaults to None. 
                Initializes to 'SLSQP'.
            opt_options (dict, optional): Dicitonary for setting the 
                optimization options. Defaults to None.
        """
        super().__init__(fi)
        self.epsilon = np.finfo(float).eps

        if opt_options is None:
            self.opt_options = {'maxiter': 100, 'disp': True, \
                         'iprint': 2, 'ftol': 1e-9}
        
        self.reinitialize_opt(
            boundaries=boundaries,
            wd = wd,
            ws = ws,
            freq = freq,
            AEP_initial=AEP_initial,
            x0=x0,
            bnds=bnds,
            min_dist=min_dist,
            opt_method=opt_method,
            opt_options=opt_options
        )

    # Private methods

    def _AEP_layout_opt(self, locs):
        self._change_coordinates(
            self._unnorm(locs, self.bndx_min, self.bndx_max))
        AEP_sum = self._AEP_loop_wd()
        return -1*AEP_sum/self.AEP_initial

    def _AEP_single_wd(self, wd, ws):
        self.fi.reinitialize_flow_field(
            wind_direction=wd, wind_speed=ws)
        self.fi.calculate_wake()

        turb_powers = [turbine.power for turbine in \
            self.fi.floris.farm.turbines]
        return np.sum(turb_powers)*self.freq*8760

    def _AEP_loop_wd(self):
        AEP_sum = 0

        for i in range(len(self.wd)):
            self.fi.reinitialize_flow_field(
                wind_direction=self.wd[i], wind_speed=self.ws[i])
            self.fi.calculate_wake()

            AEP_sum = AEP_sum + self.fi.get_farm_power()*self.freq[i]*8760
        return AEP_sum

    def _change_coordinates(self, locs):
        # Parse the layout coordinates
        layout_x = locs[0:self.nturbs]
        layout_y = locs[self.nturbs:2*self.nturbs]
        layout_array = [layout_x, layout_y]

        # Update the turbine map in floris
        self.fi.reinitialize_flow_field(layout_array=layout_array)

    def _space_constraint(self, x_in, min_dist):
        x = np.nan_to_num(x_in[0:self.nturbs])
        y = np.nan_to_num(x_in[self.nturbs:])

        dist = [np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2) \
                for i in range(self.nturbs) \
                for j in range(self.nturbs) if i != j]

        # dist = []
        # for i in range(self.nturbs):
        #     for j in range(self.nturbs):
        #         if i != j:
        #             dist.append(np.sqrt( (x[i]-x[j])**2 + (y[i]-y[j])**2))
                    
        return np.min(dist) - self._norm(min_dist, self.bndx_min, self.bndx_max)

    def _distance_from_boundaries(self, x_in, boundaries):  
        # x = self._unnorm(x_in[0:self.nturbs], self.bndx_min, self.bndx_max)
        # y = self._unnorm(x_in[self.nturbs:2*self.nturbs], \
        #                  self.bndy_min, self.bndy_max)
        x = x_in[0:self.nturbs]
        y = x_in[self.nturbs:2*self.nturbs]       
            
        dist_out = []

        for k in range(self.nturbs):
            dist = []
            in_poly = self._point_inside_polygon(x[k],y[k],boundaries)

            for i in range(len(boundaries)):
                boundaries = np.array(boundaries)
                p1 = boundaries[i]
                if i == len(boundaries)-1:
                    p2 = boundaries[0]
                else:
                    p2 = boundaries[i+1]

                px = p2[0] - p1[0]
                py = p2[1] - p1[1] 
                norm = px*px + py*py

                u = ((x[k] - boundaries[i][0])*px + \
                     (y[k] - boundaries[i][1])*py)/float(norm)

                if u <= 0:
                    xx = p1[0]
                    yy = p1[1]
                elif u >=1:
                    xx = p2[0]
                    yy = p2[1]
                else:
                    xx = p1[0] + u*px
                    yy = p1[1] + u*py

                dx = x[k] - xx
                dy = y[k] - yy
                dist.append(np.sqrt(dx*dx + dy*dy))

            dist = np.array(dist)
            if in_poly:
                dist_out.append(np.min(dist))
            else:
                dist_out.append(-np.min(dist))

        dist_out = np.array(dist_out)

        return np.min(dist_out)

    def _point_inside_polygon(self, x, y, poly):
        n = len(poly)
        inside =False

        p1x,p1y = poly[0]
        for i in range(n+1):
            p2x,p2y = poly[i % n]
            if y > min(p1y,p2y):
                if y <= max(p1y,p2y):
                    if x <= max(p1x,p2x):
                        if p1y != p2y:
                            xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x,p1y = p2x,p2y

        return inside

    def _generate_constraints(self):
        # grad_constraint1 = grad(self._space_constraint)
        # grad_constraint2 = grad(self._distance_from_boundaries)

        tmp1 = {'type': 'ineq','fun' : lambda x,*args: \
                self._space_constraint(x, self.min_dist), \
                'args':(self.min_dist,)}
        tmp2 = {'type': 'ineq','fun' : lambda x,*args: \
                self._distance_from_boundaries(x, self.boundaries_norm), \
                'args':(self.boundaries_norm,)}

        self.cons = [tmp1, tmp2]

    def _optimize(self):
        self.residual_plant = minimize(self._AEP_layout_opt,
                                self.x0,
                                method=self.opt_method,
                                bounds=self.bnds,
                                constraints=self.cons,
                                options=self.opt_options)

        opt_results = self.residual_plant.x
        
        return opt_results

    def _set_opt_bounds(self):
        self.bnds = [(0.0, 1.0) for _ in range(2*self.nturbs)]

    # Public methods

    def optimize(self):
        """
        Find optimized layout of wind turbines for power production given
        fixed atmospheric conditions (wind speed, direction, etc.).
        
        Returns:
            opt_locs (iterable): optimized locations of each turbine.
        """
        print('=====================================================')
        print('Optimizing turbine layout...')
        print('Number of parameters to optimize = ', len(self.x0))
        print('=====================================================')

        opt_locs_norm = self._optimize()

        print('Optimization complete!')

        opt_locs = [[self._unnorm(valx, self.bndx_min, self.bndx_max) \
            for valx in opt_locs_norm[0:self.nturbs]], \
            [self._unnorm(valy, self.bndy_min, self.bndy_max) \
            for valy in opt_locs_norm[self.nturbs:2*self.nturbs]]]

        return opt_locs

    def reinitialize_opt(self, boundaries=None,
                           wd = None,
                           ws = None,
                           freq = None,
                           AEP_initial = None,
                           x0=None,
                           bnds=None,
                           min_dist=None,
                           opt_method=None,
                           opt_options=None):
        """
        Reintializes parameter values for the optimization.
        
        This method reinitializes the optimization parameters and 
        bounds to the supplied values or uses what is currently stored.
        
        Args:
            boundaries (iterable): A list of pairs of floats that 
                represent the boundary's vertices. Defaults to None.
            wd (np.array, optional): An array of wind directions. 
                Defaults to None.
            ws (np.array, optional): An array of wind speeds. Defaults 
                to None.
            freq (np.array, optional): An array of wind direction 
                frequency values. Defaults to None.
            AEP_initial (float, optional): Initial Annual Energy 
                Production used in normalizing optimization. Defaults 
                to None. Initializes to the AEP of the current FLORIS 
                object.
            x0 (iterable, optional): The initial turbine locations, 
                ordered by x-coordinate and then y-coordiante 
                (ie. [x1, x2, ..., xn, y1, y2, ..., yn]). Defaults to 
                None. Initializes to the current turbine locations.
            bnds (iterable, optional): Bounds for the optimization 
                variables (pairs of min/max values for each variable). 
                Defaults to None. Initializes to the min. and max. 
                values of the boundaries iterable.
            min_dist (float, optional): The minimum distance to be 
                maitained between turbines during the optimization. 
                Defaults to None. Initializes to 2 rotor diameters.
            opt_method (str, optional): The optimization method for 
                scipy.optimize.minize to use. Defaults to None. 
                Initializes to 'SLSQP'.
            opt_options (dict, optional): Dicitonary for setting the 
                optimization options. Defaults to None.
        """
        if boundaries is not None:
            self.boundaries = boundaries
            self.bndx_min = np.min([val[0] for val in boundaries])
            self.bndy_min = np.min([val[1] for val in boundaries])
            self.bndx_max = np.max([val[0] for val in boundaries])
            self.bndy_max = np.max([val[1] for val in boundaries])
            self.boundaries_norm = [[self._norm(val[0], self.bndx_min, \
                                  self.bndx_max), self._norm(val[1], \
                                  self.bndy_min, self.bndy_max)] \
                                  for val in self.boundaries]
        if wd is not None:
            self.wd = wd
        if ws is not None:
            self.ws = ws
        if freq is not None:
            self.freq = freq
        if AEP_initial is not None:
            self.AEP_initial = AEP_initial
        else:
            self.AEP_initial = self.fi.get_farm_AEP(self.wd, \
                                                    self.ws, self.freq)
        if x0 is not None:
            self.x0 = x0
        else:
            self.x0 = [self._norm(coord.x1, self.bndx_min, self.bndx_max) for \
                coord in self.fi.floris.farm.turbine_map.coords] \
                + [self._norm(coord.x2, self.bndy_min, self.bndy_max) for \
                coord in self.fi.floris.farm.turbine_map.coords]
        if bnds is not None:
            self.bnds = bnds
        else:
            self._set_opt_bounds()
        if min_dist is not None:
            self.min_dist = min_dist
        else:
            self.min_dist = 2*self.fi.floris.farm.turbines[0].rotor_diameter
        if opt_method is not None:
            self.opt_method = opt_method
        if opt_options is not None:
            self.opt_options = opt_options

        self._generate_constraints()

    def plot_layout_opt_results(self):
        """
        Method to plot the old and new locations of the layout opitimization.
        """
        locsx_old = [self._unnorm(valx, self.bndx_min, self.bndx_max) \
                     for valx in self.x0[0:self.nturbs]]
        locsy_old = [self._unnorm(valy, self.bndy_min, self.bndy_max) \
                     for valy in self.x0[self.nturbs:2*self.nturbs]]
        locsx = [self._unnorm(valx, self.bndx_min, self.bndx_max) \
                 for valx in self.residual_plant.x[0:self.nturbs]]
        locsy = [self._unnorm(valy, self.bndy_min, self.bndy_max) \
                 for valy in self.residual_plant.x[self.nturbs:2*self.nturbs]]

        plt.figure(figsize=(9,6))
        fontsize= 16
        plt.plot(locsx_old, locsy_old, 'ob')
        plt.plot(locsx, locsy, 'or')
        # plt.title('Layout Optimization Results', fontsize=fontsize)
        plt.xlabel('x (m)', fontsize=fontsize)
        plt.ylabel('y (m)', fontsize=fontsize)
        plt.axis('equal')
        plt.grid()
        plt.tick_params(which='both', labelsize=fontsize)
        plt.legend(['Old locations', 'New locations'], loc='lower center', \
            bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=fontsize)

        verts = self.boundaries
        for i in range(len(verts)):
            if i == len(verts)-1:
                plt.plot([verts[i][0], verts[0][0]], \
                         [verts[i][1], verts[0][1]], 'b')        
            else:
                plt.plot([verts[i][0], verts[i+1][0]], \
                         [verts[i][1], verts[i+1][1]], 'b')


class LayoutHeightOptimization(LayoutOptimization):
    """
    Sub class of the 
    :py:class`floris.tools.optimization.LayoutOptimization`
    object class that performs layout and turbine height optimization.

    This optimization method aims to minimize Cost of Energy (COE) by 
    changing individual turbine locations and all turbine heights 
    across the wind farm. Note that the changing turbine height 
    applies to all turbines, i.e. although the turbine height is 
    changing, all turbines will be assigned the same turbine height.
    """

    def __init__(self, fi, boundaries,
                           height_lims,
                           wd,
                           ws,
                           freq,
                           AEP_initial,
                           COE_initial,
                           plant_kw,
                           x0=None,
                           bnds=None,
                           min_dist=None,
                           opt_method='SLSQP',
                           opt_options=None):
        """
        Instantiate LayoutHeightOptimization object and parameter 
        values.
        
        Args:
            fi (:py:class:`floris.tools.floris_utilities.FlorisInterface`): 
                Interface from FLORIS to the tools package.
            boundaries (iterable): A list of pairs of floats that 
                represent the boundary's vertices.
            height_lims (iterable): A list of the minimum and maximum 
                height limits for the optimization. Each value only 
                needs to be defined once since all the turbine heights 
                are the same (ie. [h_min, h_max]).
            wd (np.array): An array of wind directions.
            ws (np.array): An array of wind speeds.
            freq (np.array): An array of wind direction 
                frequency values.
            AEP_initial (float): Initial Annual Energy 
                Production used in normalizing optimization.
            COE_initial (float): Initial Cost of Energy used in 
                normalizing optimization.
            plant_kw (float): The rating of the entire wind plant, in 
                kW. Defaults to None.
            x0 (iterable, optional): The initial turbine locations, 
                ordered by x-coordinate and then y-coordiante 
                (ie. [x1, x2, ..., xn, y1, y2, ..., yn]), and the 
                initial turbine hub height. Defaults to None. 
                Initializes to the current turbine locations and hub 
                height.
            bnds (iterable, optional): Bounds for the optimization 
                variables (pairs of min/max values for each variable). 
                Defaults to None. Initializes to the min. and max. of 
                the boundaries iterable.
            min_dist (float, optional): The minimum distance to be 
                maitained between turbines during the optimization. 
                Defaults to None. Initializes to 2 rotor diameters.
            opt_method (str, optional): The optimization method for 
                scipy.optimize.minize to use. Defaults to None. 
                Initializes to 'SLSQP'.
            opt_options (dict, optional): Dicitonary for setting the 
                optimization options. Defaults to None.
        """
        super().__init__(fi, boundaries, wd, ws, freq, AEP_initial)
        self.epsilon = np.finfo(float).eps

        self.COE_model = BaseCOE(self)
        
        self.reinitialize_opt_height(
            boundaries=boundaries,
            height_lims=height_lims,
            wd = wd,
            ws = ws,
            freq = freq,
            AEP_initial = AEP_initial,
            COE_initial=COE_initial,
            plant_kw=plant_kw,
            x0=x0,
            bnds=bnds,
            min_dist=min_dist,
            opt_method=opt_method,
            opt_options=opt_options
        )

    # Private methods

    def _fCp_outside(self):
        pass # for futre use

    def _fCt_outside(self):
        pass # for futre use

    def _set_initial_conditions(self):
        self.x0.append( \
            self._norm(self.fi.floris.farm.turbines[0].hub_height, \
                        self.bndh_min, self.bndh_max))
    
    def _set_opt_bounds_height(self):
        self.bnds.append((0., 1.))

    def _optimize(self):
        self.residual_plant = minimize(self._COE_layout_height_opt,
                                self.x0,
                                method=self.opt_method,
                                bounds=self.bnds,
                                constraints=self.cons,
                                options=self.opt_options)

        opt_results = self.residual_plant.x
        
        return opt_results
    
    def _COE_layout_height_opt(self, opt_vars):
        locs = self._unnorm(opt_vars[0:2*self.nturbs], self.bndx_min, self.bndx_max)
        height = self._unnorm(opt_vars[-1], self.bndh_min, self.bndh_max)

        self._change_height(height)
        self._change_coordinates(locs)
        AEP_sum = self._AEP_loop_wd()
        COE = self.COE_model.COE(height, AEP_sum)

        return COE/self.COE_initial

    def _change_height(self, height):
        if isinstance(height, float) or isinstance(height, int):
            for turb in self.fi.floris.farm.turbines:
                turb.hub_height = height
        else:
            for k, turb in enumerate(self.fi.floris.farm.turbines):
                turb.hub_height = height[k]

        self.fi.reinitialize_flow_field(layout_array=((self.fi.layout_x, \
                                                       self.fi.layout_y)))

    # Public methods

    def reinitialize_opt_height(self, boundaries=None,
                           height_lims=None,
                           wd = None,
                           ws = None,
                           freq = None,
                           AEP_initial=None,
                           COE_initial = None,
                           plant_kw=None,
                           x0=None,
                           bnds=None,
                           min_dist=None,
                           opt_method=None,
                           opt_options=None):
        """
        Reintializes parameter values for the optimization.
        
        This method reinitializes the optimization parameters and 
        bounds to the supplied values or uses what is currently stored.
        
        Args:
            boundaries (iterable, optional): A list of pairs of floats 
                that represent the boundary's vertices. Defaults to 
                None.
            height_lims (iterable, optional): A list of the minimum 
                and maximum height limits for the optimization. Each 
                value only needs to be defined once since all the 
                turbine heights are the same (ie. [h_min, h_max]). 
                Defaults to None.
            wd (np.array, optional): An array of wind directions. 
                Defaults to None.
            ws (np.array, optional): An array of wind speeds. Defaults 
                to None.
            freq (np.array, optional): An array of wind direction 
                frequency values. Defaults to None.
            AEP_initial (float, optional): Initial Annual Energy 
                Production used in normalizing optimization. Defaults 
                to None.
            COE_initial (float, optional): Initial Cost of Energy used 
                in normalizing optimization. Defaults to None.
            plant_kw (float, optional): The rating of the entire wind 
                plant, in kW. Defaults to None.
            x0 (iterable, optional): The initial turbine locations, 
                ordered by x-coordinate and then y-coordiante 
                (ie. [x1, x2, ..., xn, y1, y2, ..., yn]), and the 
                initial turbine hub height. Defaults to None. 
                Initializes to the current turbine locations and hub 
                height.
            bnds (iterable, optional): Bounds for the optimization 
                variables (pairs of min/max values for each variable). 
                Defaults to None. Initializes to the min. and max. of 
                the boundaries iterable.
            min_dist (float, optional): The minimum distance to be 
                maitained between turbines during the optimization. 
                Defaults to None. Initializes to 2 rotor diameters. 
            opt_method (str, optional): The optimization method for 
                scipy.optimize.minize to use. Defaults to None. 
                Initializes to 'SLSQP'.
            opt_options (dict, optional): Dicitonary for setting the 
                optimization options. Defaults to None.
        """

        LayoutOptimization.reinitialize_opt(self, boundaries=boundaries,
                           wd=wd,
                           ws=ws,
                           freq=freq,
                           AEP_initial=AEP_initial,
                           x0=x0,
                           bnds=bnds,
                           min_dist=min_dist,
                           opt_method=opt_method,
                           opt_options=opt_options)

        if height_lims is not None:
            self.bndh_min = height_lims[0]
            self.bndh_max = height_lims[1]
        if COE_initial is not None:
            self.COE_initial = COE_initial
        if plant_kw is not None:
            self.plant_kw = plant_kw

        self._set_initial_conditions()
        self._set_opt_bounds_height()

    def optimize(self):
        """
        Find optimized layout of wind turbines and wind turbine height 
        for power production and cost of energy given fixed 
        atmospheric conditions (wind speed, direction, etc.).
        
        Returns:
            (iterable): A list containing the optimized locations of 
                each turbine and the optimized height for all turbines.
        """
        print('=====================================================')
        print('Optimizing turbine layout and height...')
        print('Number of parameters to optimize = ', len(self.x0))
        print('=====================================================')

        opt_results_norm = self._optimize()

        print('Optimization complete!')

        opt_locs = [[self._unnorm(valx, self.bndx_min, self.bndx_max) \
            for valx in opt_results_norm[0:self.nturbs]], \
            [self._unnorm(valy, self.bndy_min, self.bndy_max) \
            for valy in opt_results_norm[self.nturbs:2*self.nturbs]]]

        opt_height = [self._unnorm(opt_results_norm[-1], \
                      self.bndh_min, self.bndh_max)]

        return [opt_locs, opt_height]

    def get_farm_COE(self):
        AEP_sum = self._AEP_loop_wd()
        height = self.fi.floris.farm.turbines[0].hub_height
        return self.COE_model.COE(height, AEP_sum)


class BaseCOE():
    def __init__(self, opt_obj):
        """
        Instantiate COE model object and its parameters.
        
        Args:
            opt_obj (LayoutHeightOptimization): The optimization object.
        """
        self.opt_obj = opt_obj

    # Public methods

    def FCR(self):
        return 0.079 # % - Taken from 2016 Cost of Wind Energy Review

    def TCC(self, height):
        """
        Calcualte turbine capital costs based on varying tower cost.
        
        This method dertermines the turbine capital costs (TCC), 
        calculating the effect of varying turbine height and rotor 
        diameter on the cost of the tower. The relationship estiamted 
        the mass of steel needed for the tower from the NREL Cost and 
        Scaling Model (CSM), and then adds that to the tower cost 
        portion of the TCC. The proportion is determined from the NREL 
        2016 Cost of Wind Energy Review. A price of 3.08 $/kg is 
        assumed for the needed steel. Tower height is passed directly 
        while the turbine rotor diameter is pulled directly from the 
        turbine object within the 
        :py:class:`floris.tools.floris_utilities.FlorisInterface`:.
        
        Args:
            height (float): Turbine hub height in meters.
        
        Returns:
            float: The turbine capital cost of a wind plant in units 
            of $/kWh.
        """
        # From CSM with a fudge factor
        tower_mass = (0.2694*height*(np.pi* \
            (self.opt_obj.fi.floris.farm.turbines[0].rotor_diameter/2)**2) \
            + 1779.3)/(1.341638)

        # Combo of 2016 Cost of Wind Energy Review and CSM
        TCC = 831 + tower_mass * 3.08 \
            * self.opt_obj.nturbs / self.opt_obj.plant_kw

        return TCC

    def BOS(self):
        """
        The balance of station cost of a wind plant.
        
        The balance of station cost of wind plant as determined by a 
        constant factor. As the rating of a wind plant grows, the cost 
        of the wind plant grows as well.
        
        Returns:
            float: The balance of station cost of a wind plant in 
            units of $/kWh.
        """
        return 364. # $/kW - Taken from 2016 Cost of Wind Energy Review

    def FC(self):
        """
        The finance charge cost of a wind plant.
        
        The finance charge cost of wind plant as determined by a 
        constant factor. As the rating of a wind plant grows, the cost 
        of the wind plant grows as well.
        
        Returns:
            float: The finance charge cost of a wind plant in units 
            of $/kWh.
        """
        return 155. # $/kW - Taken from 2016 Cost of Wind Energy Review

    def O_M(self):
        """
        The operational cost of a wind plant.
        
        The operational cost of wind plant as determined by a constant 
        factor. As the rating of a wind plant grows, the cost of the 
        wind plant grows as well.
        
        Returns:
            float: The operational cost of a wind plant in units of 
            $/kWh.
        """
        return 52. # $/kW - Taken from 2016 Cost of Wind Energy Review

    def COE(self, height, AEP_sum):
        """
        The cost of energy of a wind plant.
        
        This cost of energy (COE) formulation for a wind plant varies 
        based on turbine height, rotor diameter, and total annualized 
        energy production (AEP). The components of the COE equation 
        are defined in the :py:class:`floris.tools.optimization.BaseCOE` 
        class.
        
        Args:
            height (float): The hub height of the turbines in meters 
                (all turbines are set to the same height).
            AEP_sum (float): The annualized energy production (AEP) 
                for the wind plant as calculated across the wind rose 
                in kWh.
        
        Returns:
            float: The cost of energy for a wind plant in units of 
            $/kWh.
        """
        # Comptue Cost of Energy (COE) as $/kWh for a plant
        return (self.FCR()*(self.TCC(height) + self.BOS() + self.FC()) \
                + self.O_M()) / (AEP_sum/1000/self.opt_obj.plant_kw)
        