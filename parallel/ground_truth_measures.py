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
STEP_SIZE = 14943
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

    def sample_latent(self, size=1):
        samples = np.zeros((size, self.latents_sizes.size))
        for lat_i, lat_size in enumerate(self.latents_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=size)

        return samples

    def load_image(self, index, watermark):
        img_path = os.path.join(self.img_dir, f"{index}.npy")
        image = np.load(img_path)
        image = torch.from_numpy(np.asarray(image, dtype=np.float32)).view(1, 64, 64)
        if watermark:
            image[self.water_image] = 1.0
        image = image.view(1, 1, 64, 64)
        if torch.cuda.is_available():
            image = image.cuda()
        return image

    def get_prediction(self, index, model, wm):
        image = self.load_image(index, wm)
        output = model(image)
        if self.binary:
            pred = output.data.round()
            return int(pred[0, 0])
        pred = output.data.max(1, keepdim=True)[1]
        return int(pred[0, 0])

    def get_output(self, index, model, wm):
        image = self.load_image(index, wm)
        output = model(image)
        res = output.data[0] / (torch.abs(output.data[0]).sum(-1) + 1e-10)
        return res

    def get_func(self, attribution, composite, cc, layer_names, func_type, model):
        if func_type == "bbox":
            mask = torch.zeros(64, 64)
            mask[52:, 0:17] = 1
            if torch.cuda.is_available():
                mask = mask.cuda()

            def apply_bbox_func(index, wm):
                image = self.load_image(index, wm)
                if torch.cuda.is_available():
                    image = image.cuda()
                image.requires_grad = True
                rels_masked = []
                i = 0
                if self.binary:
                    pred = 0
                else:
                    output = model(image)
                    pred = output.data.max(1, keepdim=True)[1]
                conditions = [{"linear_layers.0": [n], "y": [pred]} for n in range(6)]
                for attr in attribution.generate(
                    image,
                    conditions,
                    composite,
                    record_layer=layer_names,
                    batch_size=1,
                    exclude_parallel=False,
                    verbose=False,
                ):
                    masked = attr.heatmap * mask[None, :, :]
                    rels_masked.append(torch.sum(masked, dim=(1, 2)))
                    i += 1
                rels_masked = torch.cat(rels_masked)
                rels_masked = abs_norm(rels_masked)
                return {f"linear_layers.0_{n}": float(rels_masked[n]) for n in range(6)}

            return apply_bbox_func
        elif func_type == "simple":

            def simple_func(index, wm):
                image = self.load_image(index, wm)
                if torch.cuda.is_available():
                    image = image.cuda()
                image.requires_grad = True
                if self.binary:
                    pred = 0
                else:
                    output = model(image)
                    pred = output.data.max(1, keepdim=True)[1]
                attr = attribution(
                    image, [{"y": [pred]}], composite, record_layer=layer_names
                )
                rel_c = cc.attribute(
                    attr.relevances["linear_layers.0"], abs_norm=True
                )  #  activations
                rel_c = abs_norm(rel_c)
                return {f"linear_layers.0_{n}": float(rel_c[0][n]) for n in range(6)}

            return simple_func
        else:

            def apply_func(index, wm):
                image = self.load_image(index, wm)
                if torch.cuda.is_available():
                    image = image.cuda()
                image.requires_grad = True
                if self.binary:
                    pred = 0
                else:
                    output = model(image)
                    pred = output.data.max(1, keepdim=True)[1]
                conditions = [{"linear_layers.0": [n], "y": [pred]} for n in range(6)]
                attr = attribution(
                    image, conditions, composite, record_layer=layer_names
                )
                rel_c = cc.attribute(
                    attr.relevances["linear_layers.0"], abs_norm=True
                )  #  activations
                rel_c = abs_norm(rel_c)
                return {f"linear_layers.0_{n}": float(rel_c[0][n]) for n in range(6)}

            return apply_func

    def something_flip(self, index, apply_func):
        latents = self.index_to_latent(index)
        pred_flip = {}
        pred_flip["label"] = latents[1]
        pred_flip["pred"] = apply_func(index, False)
        wm_vals = apply_func(index, True)
        pred_flip["watermark"] = {
            f"linear_layers.0_{n}": abs(
                wm_vals[f"linear_layers.0_{n}"]
                - pred_flip["pred"][f"linear_layers.0_{n}"]
            )
            for n in range(6)
        }
        for lat in range(1, self.latents_sizes.size):
            lat_name = self.latents_names[lat]
            pred_flip[lat_name] = {f"linear_layers.0_{n}": 0.0 for n in range(6)}
            len_latent = 2 if lat_name == "shape" else self.latents_sizes[lat]
            for j in range(len_latent):
                if j != latents[lat]:
                    flip_latents = copy.deepcopy(latents)
                    flip_latents[lat] = j
                    flip_index = self.latent_to_index(flip_latents)
                    flip_pred = apply_func(flip_index, False)
                    for neur in pred_flip[lat_name].keys():
                        pred_flip[lat_name][neur] += float(
                            abs(flip_pred[neur] - pred_flip["pred"][neur])
                            / (len_latent - 1)
                        )
        return pred_flip

    def compute_multiple_neuron_flips(self, model, func_type):
        composite = EpsilonPlusFlat()
        cc = ChannelConcept()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        tdev = torch.device(device)
        attribution = CondAttribution(model, no_param_grad=True, device=tdev)
        layer_names = get_layer_names(model, [torch.nn.Conv2d, torch.nn.Linear])
        apply_func = self.get_func(
            attribution, composite, cc, layer_names, func_type, model
        )
        indices = range(0, MAX_INDEX, STEP_SIZE)
        all_inds = []
        for i in indices:
            neur_flip = self.something_flip(i, apply_func)
            all_inds.append(neur_flip)
        blub = {}
        for item in all_inds:
            for latent in item.keys():
                if isinstance(item[latent], dict) and latent != "pred":
                    if latent not in blub:
                        blub[latent] = {}
                    for neuron in item[latent].keys():
                        if neuron not in blub[latent]:
                            blub[latent][neuron] = 0.0
                        blub[latent][neuron] += item[latent][neuron]
        return blub

    def ols_values(self, model):
        """
        * For the R2 score, we fitted an ordinary least squares from the factors' deltas
        * to the deltas of the model's logits and then report the coefficient of determination.
        * Here: from latent index/value to reference score value
        """
        composite = EpsilonPlusFlat()
        cc = ChannelConcept()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        tdev = torch.device(device)
        attribution = CondAttribution(model, no_param_grad=True, device=tdev)
        layer_names = get_layer_names(model, [torch.nn.Conv2d, torch.nn.Linear])

        def apply_func(index, wm):
            image = self.load_image(index, wm)
            if torch.cuda.is_available():
                image = image.cuda()
            image.requires_grad = True
            attr = attribution(image, [{"y": [1]}], composite, record_layer=layer_names)
            rel_c = cc.attribute(
                attr.relevances["linear_layers.0"], abs_norm=True
            )  #  activations
            # rel_c = abs_norm(rel_c)
            return rel_c[0].tolist()  # [float(rel_c[0][n]) for n in range(6)]

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
        for latent in range(6):
            results.append([])
            for neuron in range(6):
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
        index_results = np.zeros((len(indices), 6, 6))
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
