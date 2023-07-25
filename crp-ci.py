import numpy as np
import math
import pandas as pd
import warnings
from scipy import stats
import numpy as np
import sys

from torch import nn 

from zennit.canonizers import SequentialMergeBatchNorm
from zennit.composites import EpsilonPlusFlat
from crp.concepts import ChannelConcept
from crp.attribution import CondAttribution
from crp.visualization import FeatureVisualization
from crp.helper import get_layer_names
from torch.autograd import Variable

from tigramite.independence_tests.independence_tests_base import CondIndTest


class CrpCI(CondIndTest):
    """Test conditional independence of relevances of neurons produces by CRP

    Parameters
    ----------
    **kwargs :
        Arguments passed on to parent class CondIndTest.
    """

    @property
    def measure(self):
        """
        Concrete property to return the measure of the independence test
        """
        return self._measure

    def __init__(self, **kwargs):
        # Setup the member variables
        self._measure = "crp_ci"
        self.two_sided = False
        self.residual_based = False
        self.recycle_residuals = False
        self.composite = EpsilonPlusFlat()
        self.cc = ChannelConcept()
        self.layer_names = None
        self.layer_map = None

        CondIndTest.__init__(self, **kwargs)

    def set_attribution(self, model: nn.Module):
        self.layer_names = get_layer_names(model, [nn.Conv2d, nn.Linear])
        self.layer_map = {layer : self.cc for layer in self.layer_names }
        self.attribution = CondAttribution(model, no_param_grad=True)


    def set_dataframe(self, dataframe):
        """Initialize and check the dataframe.

        Parameters
        ----------
        dataframe : data object
            Set tigramite dataframe object. It must have the attributes
            dataframe.values yielding a numpy array of shape (observations T,
            variables N) and optionally a mask of the same shape and a missing
            values flag.

        """
        self.dataframe = dataframe

        if self.mask_type is not None:
            if dataframe.mask is None:
                raise ValueError("mask_type is not None, but no mask in dataframe.")
            dataframe._check_mask(dataframe.mask)

        if dataframe.data_type is None:
            raise ValueError("data_type cannot be None for RegressionCI.")
        dataframe._check_mask(dataframe.data_type, check_data_type=True)

    # @jit(forceobj=True)
    def get_dependence_measure(self, array, xyz, data_type):
        pass

    def get_measure(self, X, Y, Z=None, tau_max=0, data_type=None):
        """Estimate dependence measure.

         Calls the dependence measure function. The child classes must specify
         a function get_dependence_measure.

         Parameters
         ----------
         X, Y [, Z] : list of tuples
             X,Y,Z are of the form [(var, -tau)], where var specifies the
             variable index and tau the time lag.

         tau_max : int, optional (default: 0)
             Maximum time lag. This may be used to make sure that estimates for
             different lags in X, Z, all have the same sample size.

        data_type : array-like
             Binary data array of same shape as array which describes whether
             individual samples in a variable (or all samples) are continuous
             or discrete: 0s for continuous variables and 1s for discrete variables.


         Returns
         -------
         val : float
             The test statistic value.

        """
        # Make the array
        array, xyz, (X, Y, Z), _ = self._get_array(X=X, Y=Y, Z=Z, tau_max=tau_max)
        D, T = array.shape
        # Check it is valid
        if np.isnan(array).sum() != 0:
            raise ValueError("nans in the array!")
        # Return the dependence measure
        return self._get_dependence_measure_recycle(X, Y, Z, xyz, array)
