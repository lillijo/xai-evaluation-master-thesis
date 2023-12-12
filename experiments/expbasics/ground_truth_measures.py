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

MAX_INDEX = 491520
STEP_SIZE = 7069
LATSIZE = [2, 2, 6, 40, 32, 32]


class GroundTruthMeasures:
    def __init__(self, binary=False, img_path="../dsprites-dataset/images/") -> None:
        self.img_dir = img_path
        self.water_image = np.load("watermark.npy")
        self.binary = binary
        with open("labels.pickle", "rb") as f:
            labels = pickle.load(f)
            self.labels = labels
        with open("metadata.pickle", "rb") as mf:
            metadata = pickle.load(mf)
            self.metadata = metadata
            self.latents_sizes = np.array(metadata["latents_sizes"])
            self.latents_bases = np.concatenate(
                (
                    self.latents_sizes[::-1].cumprod()[::-1][1:],
                    np.array(
                        [
                            1,
                        ]
                    ),
                )
            )
            self.latents_names = [i.decode("ascii") for i in metadata["latents_names"]]

    def latent_to_index(self, latents):
        return np.dot(latents, self.latents_bases).astype(int)

    def index_to_latent(self, index):
        return copy.deepcopy(self.labels[index])

    def load_image(self, index, watermark):
        img_path = os.path.join(self.img_dir, f"{index}.npy")
        image = np.load(img_path)
        image = torch.from_numpy(np.asarray(image, dtype=np.float32)).view(1, 64, 64)
        if watermark:
            image[self.water_image] = 1.0
        image = image.view(1, 1, 64, 64)
        if torch.cuda.is_available():
            image = image.cuda()
        image.requires_grad = True
        return image

    def get_output(self, index, model, wm):
        image = self.load_image(index, wm)
        output = model(image)
        res = output.data[0] / (torch.abs(output.data[0]).sum(-1) + 1e-10)
        return res

    def get_value_computer(self, attribution, composite, layer_name, cc, func_type):
        if func_type == "default_relevance":
            # return normed relevances of given layer
            def default_relevance(index: int, wm: bool)-> List[float]:
                image = self.load_image(index, wm)
                attr = attribution(
                    image, [{}], composite, start_layer=layer_name  # , init_rel=act
                )
                rel_c= cc.attribute(attr.relevances[layer_name], abs_norm=True)
                return rel_c[0].tolist()

            return default_relevance

        else:
            # return activations of given layer
            def apply_func(index: int, wm: bool)-> List[float]:
                image = self.load_image(index, wm)
                attr = attribution(image, [{}], composite, start_layer=layer_name)
                rel_c = cc.attribute(attr.activations[layer_name])
                return rel_c[0].tolist()

                pass

            return apply_func

    def ols_values(self, model, layer_name="linear_layers.0"):
        """
        * This is just the computation of lots of values when intervening on the generating factors
        """
        model.eval()
        composite = EpsilonPlusFlat()
        cc = ChannelConcept()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        tdev = torch.device(device)
        attribution = CondAttribution(model, no_param_grad=True, device=tdev)
        #layer_names = get_layer_names(model, [torch.nn.Conv2d, torch.nn.Linear])
        apply_func = self.get_value_computer(attribution, composite, layer_name, cc, "normal")

        indices = range(0, MAX_INDEX, STEP_SIZE)
        everything = []
        for index in indices:
            latents = self.index_to_latent(index)
            original_latents = apply_func(index, False)
            with_wm = apply_func(index, True)
            everything.append([0, 0, original_latents, True, index])
            everything.append([0, 1, with_wm, False, index])

            for lat in range(1, self.latents_sizes.size):
                lat_name = self.latents_names[lat]
                len_latent = 2 if (lat_name == "shape") else self.latents_sizes[lat]
                for j in range(len_latent):
                    if j != latents[lat]:
                        flip_latents = copy.deepcopy(latents)
                        flip_latents[lat] = j
                        flip_index = self.latent_to_index(flip_latents)
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

    def ols_prediction_values(self, model):
        """
        * SAME FOR MODELS PREDICTION
        """

        indices = range(0, MAX_INDEX, STEP_SIZE)
        everything = []
        for index in indices:
            latents = self.index_to_latent(index)
            original_output = self.get_output(index, model, False)
            with_wm_output = self.get_output(index, model, True)
            everything.append([0, 0, original_output, True, index])
            everything.append([0, 1, with_wm_output, False, index])

            for lat in range(1, self.latents_sizes.size):
                lat_name = self.latents_names[lat]
                len_latent = 2 if (lat_name == "shape") else self.latents_sizes[lat]
                for j in range(len_latent):
                    if j != latents[lat]:
                        flip_latents = copy.deepcopy(latents)
                        flip_latents[lat] = j
                        flip_index = self.latent_to_index(flip_latents)
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
                    ndiff = np.sum(np.abs(np.array(o_neuron[0] - other))) / 2
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

    def heatmaps(self, model, bias, index):
        """
        * Calculate heatmap for each neuron in each bias
        """
        image = self.load_image(index, False)
        wm_image = self.load_image(index, True)

        image.requires_grad = True
        wm_image.requires_grad = True

        composite = EpsilonPlusFlat()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        tdev = torch.device(device)
        attribution = CondAttribution(model, no_param_grad=True, device=tdev)
        layer_names = get_layer_names(model, [torch.nn.Conv2d, torch.nn.Linear])
        image_path = "heatmaps/image"
        conditions = [{"linear_layers.0": [neuron], "y": [0]} for neuron in range(6)]

        heatmaps, _, _, _ = attribution(
            image,
            conditions,
            composite,
            record_layer=layer_names,
            exclude_parallel=False,
        )
        heatmapswm, _, _, _ = attribution(
            wm_image,
            conditions,
            composite,
            record_layer=layer_names,
            exclude_parallel=False,
        )
        image_heatmaps = torch.cat((heatmaps, heatmapswm))
        vmin = min([heatmaps.min(), heatmapswm.min()])
        vmax = max([heatmaps.max(), heatmapswm.max()])
        grid = {}
        for h in range(0, 6):
            grid[f"neuron {h}"] = [heatmaps[h], heatmapswm[h]]
        img = imgify(
            image_heatmaps,
            vmax=vmax,
            vmin=vmin,
            # symmetric=True,
            grid=(len(heatmaps), 2),  # type: ignore
            padding=True,
        )
        img_name = f"{image_path}_{bias}_{index}"
        img.save(f"{img_name}.png")
        with open(f"{img_name}.pickle", "wb") as handle:
            pickle.dump(grid, handle)
        return img_name, [float(vmin), float(vmax)]
