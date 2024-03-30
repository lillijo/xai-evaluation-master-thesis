import numpy as np

from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.gpdc_torch import GPDCtorch
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.independence_tests.robust_parcorr import RobustParCorr
from tigramite.independence_tests.gsquared import Gsquared
from tigramite.independence_tests.cmisymb import CMIsymb
from tigramite.independence_tests.regressionCI import RegressionCI

from cmiknnmixed import CMIknnMixed


def remove_empty(
    data, all_var_names, types, with_type=True, delete_constant=True, verbose=False
):
    empty_vars = []
    var_names = all_var_names
    if delete_constant:
        for var in range(len(all_var_names)):
            not_constant = np.where(data[:, var] != data[0, var])[0].shape[0]
            if not_constant < 5:
                empty_vars.append(var)
            elif all_var_names[var][0:4] != "fact":
                X = data[:, var]
                data[:, var] = (X - X.mean()) / (X.std())
        var_names = np.delete(all_var_names, empty_vars)
        data = np.delete(data, empty_vars, axis=1)
        types = np.delete(types, empty_vars)
    # sanity test that variable names are correct
    if verbose:
        print(
            f"all variables: {all_var_names.shape},\n non-constant variables: {var_names.shape},\
                \n shape of dataset: {data.shape} \n var names = {var_names}"
        )
    if with_type:
        data_type = np.zeros(data.shape, dtype="int")
        data_type[:, 0 : types.shape[0]] = types
        dataframe = pp.DataFrame(data, var_names=var_names, data_type=data_type)
    else:
        dataframe = pp.DataFrame(data, var_names=var_names)
    return dataframe, var_names


def causal_discovery(dataframe, test="RobustParCorr", link_assumptions=None):
    ci_test = RobustParCorr(significance="analytic")  # With Data Type
    if test == "CMIknn":
        ci_test = CMIknn(significance="fixed_thres", fixed_thres=0.1)
    elif test == "GPDCtorch":  # No Data Type
        ci_test = GPDCtorch(significance="analytic")
    elif test == "RegressionCI":  # With Data Type
        ci_test = RegressionCI(significance="fixed_thres", fixed_thres=0.6)
    elif test == "CMIsymb":
        ci_test = CMIsymb(significance="fixed_thres", fixed_thres=0.4)
    elif test == "CMIknnMixed":  # With Data Type
        ci_test = CMIknnMixed(
            significance="fixed_thres",
            knn=0.1,
            estimator="MS", # MS MSinf
            fixed_thres=0.03,
        )  # significance='fixed_thres', fixed_thres=0.1
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ci_test, verbosity=0)

    if link_assumptions:
        results = pcmci.run_pcmciplus(
            tau_max=0, pc_alpha=0.001, link_assumptions=link_assumptions
        )
    else:
        results = pcmci.run_pcmciplus(tau_max=0, pc_alpha=0.001)
    # results = pcmci.run_pcalg_non_timeseries_data()
    return results

