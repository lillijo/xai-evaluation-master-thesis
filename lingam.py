import numpy as np
from tigramite.independence_tests.gpdc_torch import GPDCtorch

def lingam(data, i, j):
    """Performs bivariate LiNGAM causality test.

    The bivariate LiNGAM model assumes linear dependencies and that
    either X or eta^Y is non-Gaussian. Here we also assume
    no common drivers and that i and j are dependent which needs to
    be tested with a correlation test beforehand. The independence
    test is done with distance correlation (tigramite package).   
    
    """

    def indep_test(one, two):
        ind_test = GPDCtorch()

        array = np.vstack((one, two))
        xyz = np.array([0,1])

        dim, n = array.shape
        value = ind_test.get_dependence_measure(array, xyz)
        pval = ind_test.get_analytic_significance(value, T=n, dim=dim, xyz=xyz)

        return pval

    x = data[:, i].reshape(-1, 1)
    y = data[:, j].reshape(-1, 1)

    # Test causal model x --> y
    beta_hat_y = np.linalg.lstsq(x, y, rcond=None)[0]
    yresid = y - np.dot(x, beta_hat_y)
    pval_xy = indep_test(x.flatten(), yresid.flatten())

    # Test causal model y --> x
    beta_hat_x = np.linalg.lstsq(y, x, rcond=None)[0]
    xresid = x - np.dot(y, beta_hat_x)
    pval_yx = indep_test(y.flatten(), xresid.flatten())

    if pval_xy >= pval_yx:
        return 1
    else:
        return -1

    return pval, value

def make_lingam(results, data, var_names):

    graph_effects = results['graph'].copy()

    # LiNGAM on contemporaneous pairs
    for i in range(data.shape[1]):
        for j in range(i+1, data.shape[1]):
            # LiNGAM
            if results['graph'][i, j, 0] == 'o-o' or results['graph'][i, j, 0] == 'x-x':
                lingam_result = lingam(data[:1000], i, j)
                if lingam_result == 1:
                    print("LiNGAM on contemp pair %s --> %s" % (var_names[i], var_names[j]))
                    graph_effects[i,j,0] = '-->'
                    graph_effects[j,i,0] = '<--'
                else:
                    print("LiNGAM on contemp pair %s --> %s" % (var_names[j], var_names[i]))
                    graph_effects[i,j,0] = '<--'
                    graph_effects[j,i,0] = '-->'
    return graph_effects