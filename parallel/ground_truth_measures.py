from typing import List
import numpy as np
import torch
import copy
import numpy as np
import torch
import os
import pickle

from zennit.composites import EpsilonPlusFlat
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names, abs_norm
from crp.attribution import CondAttribution
from crp.image import imgify

from biased_noisy_dataset import BiasedNoisyDataset
from network import ShapeConvolutionalNeuralNetwork

MAX_INDEX = 491520
STEP_SIZE = 13267 # 1033, 2011, 2777, 5381, 7069, 13267, 18181
LATSIZE = [2, 2, 6, 40, 32, 32]


class GroundTruthMeasures:
    def __init__(self, dataset: BiasedNoisyDataset) -> None:
        self.dataset = dataset

    def get_output(self, index, model, wm):
        image = self.dataset.load_image_wm(index, wm)
        output = model(image)
        res = output.data[0] / (torch.abs(output.data[0]).sum(-1) + 1e-10)
        return res

    def get_value_computer(self, attribution, composite, layer_name, cc, func_type):
        if func_type == "default_relevance":
            # return normed relevances of given layer
            def default_relevance(index: int, wm: bool) -> List[float]:
                image = self.dataset.load_image_wm(index, wm)
                attr = attribution(
                    image, [{}], composite, start_layer=layer_name  # , init_rel=act
                )
                rel_c = cc.attribute(attr.relevances[layer_name], abs_norm=True)
                return rel_c[0].tolist()

            return default_relevance

        else:
            # return activations of given layer
            def apply_func(index: int, wm: bool) -> List[float]:
                image = self.dataset.load_image_wm(index, wm)
                attr = attribution(image, [{}], composite, start_layer=layer_name)
                rel_c = cc.attribute(attr.activations[layer_name])
                return rel_c[0].tolist()

                pass

            return apply_func

    def intervened_attributions(self, model: ShapeConvolutionalNeuralNetwork, layer_name="linear_layers.0"):
        """
        * This is just the computation of lots of values when intervening on the generating factors
        """
        model.eval()
        composite = EpsilonPlusFlat()
        cc = ChannelConcept()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        tdev = torch.device(device)
        attribution = CondAttribution(model, no_param_grad=True, device=tdev)
        # layer_names = get_layer_names(model, [torch.nn.Conv2d, torch.nn.Linear])
        apply_func = self.get_value_computer(
            attribution, composite, layer_name, cc, "default_relevance"
        )

        indices = range(0, MAX_INDEX, STEP_SIZE)
        everything = []
        for index in indices:
            latents = self.dataset.index_to_latent(index)
            original_latents = apply_func(index, False)
            with_wm = apply_func(index, True)
            everything.append([0, 0, original_latents, True, index])
            everything.append([0, 1, with_wm, False, index])

            for lat in range(1, self.dataset.latents_sizes.size):
                lat_name = self.dataset.latents_names[lat]
                len_latent = (
                    2 if (lat_name == "shape") else self.dataset.latents_sizes[lat]
                )
                for j in range(len_latent):
                    if j != latents[lat]:
                        flip_latents = copy.deepcopy(latents)
                        flip_latents[lat] = j
                        flip_index = self.dataset.latent_to_index(flip_latents)
                        flip_pred = apply_func(flip_index, False)
                        everything.append([lat, j, flip_pred, False, index])
                    else:
                        everything.append([lat, j, original_latents, True, index])
        return everything

    def ordinary_least_squares(self, everything):
        results = []
        n_neur = len(everything[0][2])
        for latent in range(6):
            results.append([])
            for neuron in range(n_neur):
                vals = list(filter(lambda x: x[0] == latent, everything))
                predict = [x[1] for x in vals]
                actual = [x[2][neuron] for x in vals]
                corr_matrix = np.corrcoef(actual, predict)
                corr = corr_matrix[0, 1]
                R_sq = corr**2
                if np.isnan(R_sq):
                    R_sq = 0.0
                results[latent].append(R_sq)
        return results

    def mean_logit_change(self, everything):
        indices = range(0, MAX_INDEX, STEP_SIZE)
        n_neur = len(everything[0][2])
        index_results = np.zeros((len(indices), 6, n_neur))
        for count, index in enumerate(indices):
            vals_index = list(filter(lambda x: x[4] == index, everything))
            for latent in range(6):
                vals = list(filter(lambda x: x[0] == latent, vals_index))
                original_v = list(filter(lambda x: x[3], vals))
                changed_v = list(filter(lambda x: not x[3], vals))
                o_neuron = np.array([x[2] for x in original_v])
                c_neurons = np.array([x[2] for x in changed_v])
                mean_change = 0
                for o, other in enumerate(c_neurons):
                    ndiff = np.abs(o_neuron[0] - other)
                    mean_change += ndiff
                mean_change = mean_change / len(c_neurons)
                index_results[count][latent] = mean_change
        results = np.sum(index_results, 0) / len(indices)
        return results

    def intervened_predictions(self, model):
        """
        * SAME FOR MODELS PREDICTION
        """

        indices = range(0, MAX_INDEX, STEP_SIZE)
        everything = []
        for index in indices:
            latents = self.dataset.index_to_latent(index)
            original_output = self.get_output(index, model, False)
            with_wm_output = self.get_output(index, model, True)
            everything.append([0, 0, original_output, True, index])
            everything.append([0, 1, with_wm_output, False, index])

            for lat in range(1, self.dataset.latents_sizes.size):
                lat_name = self.dataset.latents_names[lat]
                len_latent = (
                    2 if (lat_name == "shape") else self.dataset.latents_sizes[lat]
                )
                for j in range(len_latent):
                    if j != latents[lat]:
                        flip_latents = copy.deepcopy(latents)
                        flip_latents[lat] = j
                        flip_index = self.dataset.latent_to_index(flip_latents)
                        flip_pred = self.get_output(flip_index, model, False)
                        everything.append([lat, j, flip_pred, False, index])
                    else:
                        everything.append([lat, j, original_output, True, index])
        return everything

    def ordinary_least_squares_prediction(self, everything):
        results = []
        for latent in range(6):
            results.append([])
            vals = list(filter(lambda x: x[0] == latent, everything))
            predict = torch.tensor([torch.tensor(x[1]) for x in vals])
            actual = torch.tensor([x[2].max(0, keepdim=True).indices for x in vals])
            corr_matrix = np.corrcoef(actual, predict)
            corr = corr_matrix[0, 1]
            R_sq = corr**2
            if np.isnan(R_sq):
                R_sq = 0.0
            results[latent].append(R_sq)
        return results

    def mean_logit_change_prediction(self, everything):
        """
        * same as for neurons but with output value
        """
        indices = range(0, MAX_INDEX, STEP_SIZE)
        index_results = np.zeros((len(indices), 6))
        for count, index in enumerate(indices):
            vals_index = list(filter(lambda x: x[4] == index, everything))
            for latent in range(6):
                vals = list(filter(lambda x: x[0] == latent, vals_index))
                original_v = list(filter(lambda x: x[3], vals))
                changed_v = list(filter(lambda x: not x[3], vals))
                o_neuron = [x[2] for x in original_v]
                c_neurons = [x[2] for x in changed_v]
                mean_change = 0
                for other in c_neurons:
                    ndiff = np.sum(np.abs(np.array(o_neuron[0].cpu() - other.cpu()))) / 2
                    mean_change += ndiff
                mean_change = mean_change / (len(c_neurons))
                index_results[count][latent] = mean_change
        results = np.sum(index_results, 0) / len(indices)
        return results

    def prediction_flip(self, everything):
        indices = range(0, MAX_INDEX, STEP_SIZE)
        index_results = np.zeros((len(indices), 6))
        for count, index in enumerate(indices):
            vals_index = list(filter(lambda x: x[4] == index, everything))
            for latent in range(6):
                vals = list(filter(lambda x: x[0] == latent, vals_index))
                original_v = list(filter(lambda x: x[3], vals))
                changed_v = list(filter(lambda x: not x[3], vals))
                og_prediction = [x[2].max(0, keepdim=True).indices for x in original_v]
                ch_prediction = [x[2].max(0, keepdim=True).indices for x in changed_v]
                num_changed = 0
                for i in range(len(ch_prediction)):
                    if ch_prediction[i][0] != og_prediction[0][0]:
                        num_changed += 1
                mean_change = num_changed / (len(ch_prediction))
                index_results[count][latent] = mean_change
        results = np.sum(index_results, 0) / len(indices)
        return results
