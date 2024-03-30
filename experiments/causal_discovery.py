from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import torch

from tigramite import data_processing as pp

# from tigramite.toymodels import structural_causal_processes
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.gpdc_torch import GPDCtorch
from tigramite.independence_tests.cmiknn import CMIknn

# from tigramite.causal_effects import CausalEffects
from tigramite.independence_tests.robust_parcorr import RobustParCorr
from tigramite.independence_tests.gsquared import Gsquared
from tigramite.independence_tests.cmisymb import CMIsymb
from tigramite.independence_tests.regressionCI import RegressionCI

from crp_attribution import CRPAttribution
from wdsprites_dataset import DSPritesDataset
from network import ShapeConvolutionalNeuralNetwork
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


class CausalDiscovery:
    def __init__(
        self,
        dataset: DSPritesDataset,
        model: ShapeConvolutionalNeuralNetwork,
        crpattribution: CRPAttribution,
    ) -> None:
        self.ds = dataset
        self.model = model
        self.crpatrr = crpattribution
        self.layers = [
            ["factors", ["shape", "scale", "rot", "posX", "posY", "watermark"]],
            # ["convolutional_layers.0", range(6)],
            # ["convolutional_layers.3", [5]],
            ["convolutional_layers.6", range(6)],
            ["linear_layers.0", range(6)],
            ["linear_layers.2", range(2)],
            # ["prediction",  ["rectangle", "ellipse", "heart"]] #  ["class"]],  #
        ]  # , ["prediction", range(3)]

    def change_layers(self, new_l):
        self.layers = new_l

    def draw_complete_ref_score_values(self, size, layers, shape=None):
        variables = []
        indices = []
        while len(variables) < size:
            # test_unbiased, train, no_watermark
            index = np.random.randint(0, len(self.ds))
            indices.append(index)
            latents, watermark, offset = self.ds.get_item_info(index)
            # only specific shape
            img, label = self.ds[index]
            if shape is not None and label != shape:
                continue
            sample = img.view(1, 1, 64, 64)
            sample.requires_grad = True
            output = self.model(sample)
            pred = int(output.data.max(1)[1][0])
            in_variables = []
            for x in layers:
                cond_layer = x[0]
                if cond_layer == "factors":
                    res = latents.tolist() + [int(watermark)]
                elif cond_layer == "prediction":
                    if len(x[1]) == 1:
                        res = [pred]
                    else:
                        res = output.data[0].tolist()
                else:
                    neurons = x[1]
                    res = self.crpatrr.get_reference_scores(
                        sample, pred, cond_layer, neurons
                    )
                in_variables += res

            if not np.all(np.isfinite(in_variables)):
                print(index)
            variables.append(in_variables)
        return np.array(variables, dtype=np.float64), np.array(indices, dtype=np.int64)

    def no_constants_df(self, shape=None, with_type=True):
        all_var_names = np.array(
            [f"{nam[0][0:4]}{nam[0][-1]}_{k}" for nam in self.layers for k in nam[1]]
        )
        data, indices = self.draw_complete_ref_score_values(2000, self.layers, shape)
        empty_vars = []
        var_names = all_var_names
        for var in range(len(all_var_names)):
            not_constant = np.where(data[:, var] != data[0, var])[0].shape[0]
            if not_constant < 5:
                empty_vars.append(var)
            elif all_var_names[var][0:4] != "fact":
                X = data[:, var]
                data[:, var] = (X - X.mean()) / (X.std())
        var_names = np.delete(all_var_names, empty_vars)
        data = np.delete(data, empty_vars, axis=1)
        layers = [
            [
                nam[0],
                [k for k in nam[1] if f"{nam[0][0:4]}{nam[0][-1]}_{k}" in var_names],
            ]
            for nam in self.layers
        ]
        # sanity test that variable names are correct
        print(
            f"all variables: {all_var_names.shape},\n non-constant variables: {var_names.shape},\
                \n shape of dataset: {data.shape} \n new layers: \n{layers}\n var names = {var_names}"
        )
        if with_type:
            types = np.zeros(data.shape, dtype="int")
            if layers[0][0] == "factors":
                types[:, : len(layers[0][1])] = 1
            if var_names[-1] == "predn_class":
                types[:, -1] = 1
            dataframe = pp.DataFrame(data, var_names=var_names, data_type=types)
        else:
            dataframe = pp.DataFrame(data, var_names=var_names)
        return dataframe, var_names, layers, indices

    def make_nn_link_assumptions(self, layers, same_layer=False, all_factors=False):
        link_assumptions = {}
        index = 0
        layerEnd = 0
        for l in range(len(layers)):
            previousEnd = layerEnd
            layerEnd += len(layers[l][1])
            for neuron in layers[l][1]:
                if same_layer or (all_factors and layers[l][0] == "factors"):
                    link_assumptions[index] = {}
                    for i in range(previousEnd, layerEnd):
                        if i != index:
                            link_assumptions[index][(i, 0)] = "o?o"
                            if i not in link_assumptions:
                                link_assumptions[i] = {}
                            link_assumptions[i][(index, 0)] = "o?o"
                if l + 1 < len(layers):
                    if index not in link_assumptions:
                        link_assumptions[index] = {}
                    otherneurons = len(layers[l + 1][1])
                    if all_factors and l == 0:
                        otherneurons = sum(
                            len(layers[i][1]) for i in range(1, len(layers))
                        )
                    for other_neuron in range(otherneurons):
                        othern = layerEnd + other_neuron
                        link_assumptions[index][(othern, 0)] = "<?-"
                        if othern not in link_assumptions:
                            link_assumptions[othern] = {}
                        link_assumptions[othern][(index, 0)] = "-?>"
                    index += 1
        return link_assumptions

    def causal_discovery(
        self, layers, dataframe, test="RobustParCorr", link_assum=True
    ):
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
                significance="fixed_thres", fixed_thres=0.05
            )  # significance='fixed_thres', fixed_thres=0.1

        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ci_test, verbosity=0)
        if link_assum:
            link_assumptions = self.make_nn_link_assumptions(layers, False, False)
            results = pcmci.run_pcmciplus(
                tau_max=0, pc_alpha=0.01, link_assumptions=link_assumptions
            )
        else:
            results = pcmci.run_pcmciplus(tau_max=0, pc_alpha=0.01)
        # results = pcmci.run_pcalg_non_timeseries_data()
        return results
