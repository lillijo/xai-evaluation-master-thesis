from typing import List
import numpy as np
import torch
import copy
import torch
from tqdm import tqdm

from zennit.composites import EpsilonPlusFlat
from crp.concepts import ChannelConcept
from crp.helper import get_layer_names
from crp.attribution import CondAttribution

from expbasics.biased_noisy_dataset import BiasedNoisyDataset

MAX_INDEX = 491520
STEP_SIZE = 13267  # 1033, 2011, 2777, 5381, 7069, 13267, 18181
LATSIZE = [2, 2, 6, 40, 32, 32]
LAYER_ID_MAP = {
    "convolutional_layers.0": 8,
    "convolutional_layers.3": 8,
    "convolutional_layers.6": 8,
    "linear_layers.0": 6,
    "linear_layers.2": 2,
}


class GroundTruthMeasures:
    def __init__(self, dataset: BiasedNoisyDataset, step_size=STEP_SIZE) -> None:
        self.dataset = dataset
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tdev = torch.device(self.device)
        self.max_index = MAX_INDEX
        self.step_size = step_size

    def get_output(self, index, model, wm):
        image = self.dataset.load_image_wm(index, wm)
        output = model(image)
        res = output.data[0] / (torch.abs(output.data[0]).sum(-1) + 1e-10)
        return res

    def get_value_computer(self, layer_name, model, func_type):
        model.eval()
        composite = EpsilonPlusFlat()
        cc = ChannelConcept()
        attribution = CondAttribution(model, no_param_grad=True, device=self.tdev)
        layer_names = get_layer_names(model, [torch.nn.Conv2d, torch.nn.Linear])

        def crp_wm_bbox_layer(index: int, wm: bool):
            image = self.dataset.load_image_wm(index, wm)
            _, _, offset = self.dataset.get_item_info(index)
            mask = torch.zeros(64, 64).to(self.tdev)
            mask[
                max(0, 57 + offset[0]) : max(0, 58 + offset[0]) + 5,
                max(offset[1] + 3, 0) : max(offset[1] + 4, 0) + 10,
            ] = 1
            mask_size = int(mask.sum())
            nlen = LAYER_ID_MAP[layer_name]

            output = model(image)
            pred = int(output.data.max(1)[1][0])
            conditions = [
                {layer_name: [i], "y": [pred]} for i in range(LAYER_ID_MAP[layer_name])
            ]
            if layer_name == "linear_layers.2":
                conditions = [{"y": [pred]}]

            rel_within = []
            rra = []
            rel_total = []
            for attr in attribution.generate(
                image,
                conditions,
                composite,
                record_layer=layer_names,
                verbose=False,
                batch_size=nlen,
            ):
                heatmap = attr.heatmap.abs()
                masked = heatmap * mask

                # relevance rank accuracy:
                cutoffs = torch.sort(
                    heatmap.view(nlen, -1), dim=1, descending=True
                ).values[:, mask_size]
                cutoffs = torch.where(cutoffs > 0, cutoffs, 100)
                rrank = masked >= cutoffs[:, None, None]
                rank_counts = torch.count_nonzero(rrank, dim=(1, 2))
                rra.append(rank_counts / mask_size)

                # antimasked = attr.heatmap * antimask
                rel_within.append(torch.sum(masked, dim=(1, 2)))
                rel_total.append(torch.sum(heatmap, dim=(1, 2)))
            # relevance mass accuracy: R_within / R_total
            rel_within = torch.cat(rel_within)  # abs_norm* 100
            rel_total = torch.cat(rel_total)
            rra = torch.cat(rra)
            rma = rel_within / (rel_total + 1e-10)

            return dict(
                rel_total=rel_total.tolist(),
                rel_within=rel_within.tolist(),
                rma=rma.tolist(),
                rra=rra.tolist(),
                pred=pred,
            )

        if func_type == "complete_heatmaps":

            def complete_heatmaps(index: int, wm: bool):
                image = self.dataset.load_image_wm(index, wm)
                nlen = LAYER_ID_MAP[layer_name]

                output = model(image)
                pred = int(output.data.max(1)[1][0])
                conditions = [
                    {layer_name: [i], "y": [pred]}
                    for i in range(LAYER_ID_MAP[layer_name])
                ]
                if layer_name == "linear_layers.2":
                    conditions = [{"y": [pred]}]
                heatmaps = np.zeros((nlen, 64, 64))
                for attr in attribution.generate(
                    image,
                    conditions,
                    composite,
                    record_layer=layer_names,
                    verbose=False,
                    batch_size=nlen,
                ):
                    heatmaps = attr.heatmap

                return heatmaps

            return complete_heatmaps

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
        elif func_type == "rma":
            return lambda index, wm: crp_wm_bbox_layer(index, wm)["rma"]
        elif func_type == "rra":
            return lambda index, wm: crp_wm_bbox_layer(index, wm)["rra"]
        elif func_type == "rel_within":
            return lambda index, wm: crp_wm_bbox_layer(index, wm)["rel_within"]
        elif func_type == "bbox_all":
            return crp_wm_bbox_layer
        else:
            # return activations of given layer
            def apply_func(index: int, wm: bool) -> List[float]:
                image = self.dataset.load_image_wm(index, wm)
                attr = attribution(image, [{}], composite, start_layer=layer_name)
                rel_c = cc.attribute(attr.activations[layer_name])
                return rel_c[0].tolist()

            return apply_func

    def intervened_attributions(
        self,
        model,
        layer_name="linear_layers.0",
        func_type="default_relevance",
        disable=False,
    ):
        """
        * This is just the computation of lots of values when intervening on the generating factors
        """

        apply_func = self.get_value_computer(layer_name, model, func_type)

        indices = range(0, self.max_index, self.step_size)
        everything = []
        for index in tqdm(indices, disable=disable):
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

    def bounding_box_collection(
        self,
        model,
        layer_name="linear_layers.0",
        disable=False,
    ):
        """
        * This is just the computation of lots of values when intervening on the generating factors
        """

        apply_func = self.get_value_computer(layer_name, model, func_type="bbox_all")
        indices = range(0, self.max_index, self.step_size)
        everything_rma = []
        everything_rra = []
        for index in tqdm(indices, disable=disable):
            no_wm = apply_func(index, False)
            with_wm = apply_func(index, True)
            # latent type, latent index, value, is original, index
            everything_rma.append([0, 0, no_wm["rma"], False, index, no_wm["pred"]])  # type: ignore
            everything_rma.append([0, 1, with_wm["rma"], True, index, with_wm["pred"]])  # type: ignore
            everything_rra.append([0, 0, no_wm["rra"], False, index, no_wm["pred"]])  # type: ignore
            everything_rra.append([0, 1, with_wm["rra"], True, index, with_wm["pred"]])  # type: ignore
        return everything_rma, everything_rra

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

    def mean_logit_change(self, everything, n_latents=6):
        indices = range(0, self.max_index, self.step_size)
        n_neur = len(everything[0][2])
        index_results = np.zeros((len(indices), n_latents, n_neur))
        for count, index in enumerate(indices):
            vals_index = list(filter(lambda x: x[4] == index, everything))
            for latent in range(n_latents):
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

        indices = range(0, self.max_index, self.step_size)
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
                        flip_pred = self.get_output(flip_index, model, True)
                        # latent, latent index, value, is original, index
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
            actual = torch.tensor([x[2][0] - x[2][1] for x in vals])
            # print(actual.shape, predict.shape, [(x[2][0] - x[2][1]) for x in vals])
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
        indices = range(0, self.max_index, self.step_size)
        index_results = np.zeros((len(indices), 6))
        for count, index in enumerate(indices):
            vals_index = list(filter(lambda x: x[4] == index, everything))
            for latent in range(6):
                vals = list(filter(lambda x: x[0] == latent, vals_index))
                original_v = list(filter(lambda x: x[3], vals))
                changed_v = list(filter(lambda x: not x[3], vals))
                o_neuron = [x[2].to(self.tdev) for x in original_v]
                c_neurons = [x[2].to(self.tdev) for x in changed_v]
                mean_change = 0
                for other in c_neurons:
                    ndiff = torch.sum(torch.abs(o_neuron[0] - other)) / 2
                    mean_change += ndiff
                mean_change = mean_change / (len(c_neurons))
                index_results[count][latent] = mean_change
        results = np.sum(index_results, 0) / len(indices)
        return results

    def prediction_flip(self, everything):
        indices = range(0, self.max_index, self.step_size)
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
