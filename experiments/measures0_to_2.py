from math import isnan
import pickle
import numpy as np
import torch
from tqdm import tqdm
import json
import copy
from sklearn.metrics import matthews_corrcoef, normalized_mutual_info_score
from torch.utils.data import DataLoader, Subset
from expbasics.biased_noisy_dataset import BiasedNoisyDataset
from expbasics.crp_attribution import CRPAttribution
from expbasics.network import load_model


def to_name(b, i):
    return "b{}-i{}".format(
        str(round(b, 2)).replace(".", "_"),
        str(i),
    )


MAX_INDEX = 491519  # 491520 is true size, but not including last
SAMPLE_SET_SIZE = 1000
LATSIZE = [2, 2, 6, 40, 32, 32]
SEED = 42
LAYER_ID_MAP = {
    "convolutional_layers.0": 8,
    "convolutional_layers.3": 8,
    "convolutional_layers.6": 8,
    "linear_layers.0": 6,
    "linear_layers.2": 2,
}
NAME = "../clustermodels/final"  # "../clustermodels/model_seeded"  #
BIASES = list(np.round(np.linspace(0, 1, 51), 3))
# list(np.round(np.linspace(0, 1, 11), 3))
# list(np.round(np.linspace(0, 1, 51), 3))
# list(np.round(np.linspace(0, 1, 21), 3))  #
ITERATIONS = list(range(16))  # list(range(10))  #

ALL_MEASURES = [
    "m1_pf",
    "m1_mlc_euclid_norm",
    "m1_mlc_euclid_abs_norm",
    "m1_mlc_euclid",
    "m2_crv",
    "m2_mac_absolute",
    "m2_mac_absolute_weigh",
    "m2_mac_euclid",
    "m2_mac_euclid_abs_norm",
    "m2_mac_euclid_norm",
    "m2_mac_euclid_weigh_abs_norm",
    "m2_mac_euclid_weigh_norm",
    "m2_mac_euclid_weigh",
    "m2_relative_mask",
    "m2_relative_mask_weigh",
    "m2_rma",
    "m2_rma_val",
    "m2_rma_euclid",
    "m2_rra",
    "m2_rra_weigh",
    "m2_rra_euclid",
    "m2_pg",
    "m2_pg_non_empty",
    "m2_pg_weigh",
    "m2_pg_val",
    "m2_kernel",
    "m0_rho",
    "m0_mi",
    "m0_phi",
]


class AllMeasures:
    def __init__(self, img_path="", sample_set_size=SAMPLE_SET_SIZE) -> None:
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tdev = torch.device(self.device)
        self.max_index = MAX_INDEX
        self.len_x = sample_set_size
        self.ds = BiasedNoisyDataset(0, 0.5, False, img_path=img_path)
        self.rng = np.random.default_rng()  # seed=SEED

    def mean_absolute_change(self, v1, v2):
        if isinstance(v1, int):
            return abs(v1 - v2)
        return float(torch.sum(torch.abs(v1 - v2))) / v1.shape[0]

    def diff_normed(self, v1, v2):
        return float(
            torch.sum(
                torch.abs(
                    (v1 / (torch.norm(v1) + 1e-10)) - (v2 / (torch.norm(v2) + 1e-10))
                )
            )
        )

    def hm_diff1(self, h0, h1, r0, r1):
        return float(
            torch.sum(torch.abs((h1 * r1[:, None, None] - h0 * r0[:, None, None])))
        )

    def euclidean_distance(self, h0, h1, normed="norm"):
        h0 = torch.flatten(h0)
        h1 = torch.flatten(h1)
        if normed == "norm":
            if torch.any(h0 != 0):
                h0 = h0 / (torch.norm(h0))
            if torch.any(h1 != 0):
                h1 = h1 / (torch.norm(h1))
        elif normed == "abs_norm":
            if torch.any(h0 != 0):
                h0 = h0 / h0.abs().max()
            if torch.any(h1 != 0):
                h1 = h1 / h1.abs().max()
        return float(torch.sqrt(torch.sum(torch.square(h1 - h0))))

    def cosine_similarity(self, h0, h1):
        return float(
            1
            - torch.abs(
                torch.nn.functional.cosine_similarity(
                    torch.flatten(h1),
                    torch.flatten(h0),
                    dim=0,
                )
            )
        )

    def earth_mover_distance(self, h0, h1, w=None):
        pass

    def heatmaps(self, image, wm_mask, layer_name, crpa: CRPAttribution):
        nlen = LAYER_ID_MAP[layer_name]
        conditions = [
            {layer_name: [i], "y": [1]} for i in range(LAYER_ID_MAP[layer_name])
        ]
        mask_size = int(wm_mask.sum())
        heatmaps = torch.zeros((nlen, 64, 64))
        rel_within = []
        rel_total = []
        rra = []
        pointing_game = torch.zeros(nlen)
        non_empty = 0
        for attr in crpa.attribution.generate(
            image,
            conditions,
            crpa.composite,
            # start_layer="linear_layers.2",
            record_layer=crpa.layer_names,
            verbose=False,
            batch_size=nlen,
        ):
            heatmaps = attr.heatmap
            heatmaps_abs = attr.heatmap.abs()
            masked = heatmaps_abs * wm_mask
            sorted_values = torch.sort(
                heatmaps_abs.view(nlen, -1), dim=1, descending=True
            ).values
            cutoffs = sorted_values[:, mask_size]
            cutoffs = torch.where(cutoffs > 0, cutoffs, 100)
            rrank = masked >= cutoffs[:, None, None]

            # pointing game
            largest = sorted_values[:, 0]
            non_empty = torch.count_nonzero(largest > 0)
            largest = torch.where(largest > 0, largest, 100)
            pointing_game = (torch.max(masked.view(nlen, -1)) >= largest).float()
            ## rra
            rank_counts = torch.count_nonzero(rrank, dim=(1, 2))
            rra.append(rank_counts / mask_size)

            rel_within.append(torch.sum(masked, dim=(1, 2)))
            rel_total.append(torch.sum(heatmaps_abs, dim=(1, 2)))

        rel_within = torch.cat(rel_within)
        rel_total = torch.cat(rel_total)
        rra = torch.cat(rra)
        rma = rel_within / (rel_total + 1e-10)
        return dict(
            heatmaps=heatmaps,
            rel_total=rel_total,
            rel_within=rel_within,
            rma=rma,
            rra=rra,
            pg=pointing_game,
            non_empty=non_empty,
        )

    def relative_masked_relevance(
        self, index, image, wm_mask, layer_name, crpa: CRPAttribution
    ):
        shape_mask = self.ds.load_shape_mask(index)
        nlen = LAYER_ID_MAP[layer_name]
        conditions = [{layer_name: [i]} for i in range(LAYER_ID_MAP[layer_name])]
        vals = torch.zeros(nlen)
        for attr in crpa.attribution.generate(
            image,
            conditions,
            crpa.composite,
            start_layer="linear_layers.2",
            record_layer=crpa.layer_names,
            verbose=False,
            batch_size=nlen,
        ):
            heatmaps_abs = attr.heatmap.abs()
            wm_rel = torch.sum(heatmaps_abs * wm_mask[0], dim=(1, 2))
            shape_rel = torch.sum(heatmaps_abs * shape_mask[0], dim=(1, 2))
            for i in range(nlen):
                if wm_rel[i] > (shape_rel[i] * (wm_mask.sum() / (shape_mask.sum()))):
                    vals[i] = wm_rel[i]
                else:
                    vals[i] = 0
        return vals

    def relevances(self, image, label, layer_name, crpa: CRPAttribution):
        attr = crpa.attribution(
            image,
            [{"y": [1]}],
            crpa.composite,
            record_layer=crpa.layer_names,
        )
        rel_c = crpa.cc.attribute(attr.relevances[layer_name], abs_norm=True)
        return rel_c[0]

    def measure_function(self, measure, args):
        [
            predv_0,
            predv_1,
            predi_0,
            predi_1,
            rel_0,
            rel_1,
            hm_0,
            hm_1,
            x,
            label,
            image_0,
            image_1,
            mask,
            layer_name,
            crpa,
        ] = args
        # mean logit change using absolute summed difference
        if "m1_mlc" == measure:
            return self.mean_absolute_change(predv_1[0], predv_0[0])
        # mean logit change using euclidean distance
        if "m1_mlc_euclid_norm" == measure:
            return self.euclidean_distance(predv_1, predv_0, normed="norm")
        if "m1_mlc_euclid_abs_norm" == measure:
            return self.euclidean_distance(predv_1, predv_0, normed="abs_norm")
        if "m1_mlc_euclid" == measure:
            return self.euclidean_distance(predv_1, predv_0, normed="none")
        # mean logit change using cosine distance
        """ if "m1_mlc_cosine" == measure:
            return self.cosine_similarity(predv_1[0], predv_0[0]) """
        # prediction flip
        if "m1_pf" == measure:
            return self.mean_absolute_change(predi_1, predi_0)
        # distance between vectors of all relevances
        if "m2_crv" == measure:
            # relevance vector for multiple layers
            rel_0_all = torch.cat(
                [
                    self.relevances(image_0, label, l, crpa)
                    for l in list(LAYER_ID_MAP.keys())[1:4]
                ]
            )
            rel_1_all = torch.cat(
                [
                    self.relevances(image_1, label, l, crpa)
                    for l in list(LAYER_ID_MAP.keys())[1:4]
                ]
            )
            # relevances over 3 layers, excluding first and last
            # each layers absolute values sum up to 1
            # having 3 layers, we have to divide by 6
            # because maximal difference is 2
            return float(torch.sum(torch.abs(rel_0_all - rel_1_all)) / 6)
        if "m2_mac_absolute" == measure:
            return float(torch.sum(torch.abs(hm_1["heatmaps"] - hm_0["heatmaps"])))
        if "m2_mac_absolute_weigh" == measure:
            diff = 0
            for i in range(LAYER_ID_MAP[layer_name]):
                diff += torch.sum(
                    torch.abs(
                        hm_1["heatmaps"][i] * torch.abs(rel_1[i])
                        - hm_0["heatmaps"][i] * torch.abs(rel_0[i])
                    )
                )
            return float(diff)
        # euclidean distance heatmaps, with/without weights
        if measure.startswith("m2_mac_euclid"):
            diff = 0
            norm = "none"
            if measure.endswith("abs_norm"):
                norm = "abs_norm"
            elif measure.endswith("norm"):
                norm = "norm"
            weight_1 = (
                torch.abs(rel_1)
                if "weigh" in measure
                else torch.ones(LAYER_ID_MAP[layer_name])
            )
            weight_0 = (
                torch.abs(rel_0)
                if "weigh" in measure
                else torch.ones(LAYER_ID_MAP[layer_name])
            )
            for i in range(LAYER_ID_MAP[layer_name]):
                diff += self.euclidean_distance(
                    hm_0["heatmaps"][i] * weight_0[i],
                    hm_1["heatmaps"][i] * weight_1[i],
                    normed=norm,
                )
            return float(diff)
        # cosine similarity heatmaps, with/without weights
        """ if "m2_mac_cosine" == measure:
            return self.cosine_similarity(hm_0["heatmaps"], hm_1["heatmaps"])
        if "m2_mac_cosine_weigh" == measure:
            # not valid anymore
            return self.cosine_similarity(hm_0["heatmaps"], hm_1["heatmaps"]) """
        if "m2_relative_mask" == measure:
            vals = self.relative_masked_relevance(x, image_1, mask, layer_name, crpa)
            return float(vals.sum())
        if "m2_relative_mask_weigh" == measure:
            vals = self.relative_masked_relevance(x, image_1, mask, layer_name, crpa)
            diff = 0
            for i in range(LAYER_ID_MAP[layer_name]):
                diff += vals[i] * torch.abs(rel_1[i])
            return float(diff)
        if "m2_rma" == measure:
            return self.mean_absolute_change(hm_1["rma"], hm_0["rma"])
        if "m2_rma_val" == measure:
            return float(torch.sum(hm_1["rma"]))
        if "m2_rma_euclid" == measure:
            return self.euclidean_distance(hm_1["rma"], hm_0["rma"], normed="none")
        if "m2_rra" == measure:
            return float(torch.sum(torch.abs(hm_1["rra"] - hm_0["rra"])))
        if "m2_rra_weigh" == measure:
            diff = 0
            for i in range(LAYER_ID_MAP[layer_name]):
                diff += torch.abs(hm_1["rra"][i] - hm_0["rra"][i]) * torch.abs(rel_1[i])
            return float(diff)
        if "m2_rra_euclid" == measure:
            return self.euclidean_distance(hm_1["rra"], hm_0["rra"], normed="none")
        if "m2_pg" == measure:
            return self.mean_absolute_change(hm_1["pg"], hm_0["pg"])
        if "m2_pg_non_empty" == measure:
            if hm_1["non_empty"] > 0:
                return float(
                    torch.sum(torch.abs(hm_1["pg"] - hm_0["pg"])) / hm_1["non_empty"]
                )
            return 0.0
        if "m2_pg_weigh" == measure:
            diff = 0
            for i in range(LAYER_ID_MAP[layer_name]):
                diff += torch.abs(hm_1["pg"][i] - hm_0["pg"][i]) * torch.abs(rel_1[i])
            return float(diff)
        if "m2_pg_val" == measure:
            return float(torch.sum(hm_1["pg"]) / hm_1["pg"].shape[0])
        if "m2_kernel" == measure:

            def kernel(a, b):
                return (
                    torch.nn.functional.conv2d(a, a)
                    - 2 * torch.nn.functional.conv2d(a, b)
                    + torch.nn.functional.conv2d(b, b)
                )

            batched = kernel(
                hm_1["heatmaps"].view(LAYER_ID_MAP[layer_name], 1, 64, 64),
                hm_0["heatmaps"].view(LAYER_ID_MAP[layer_name], 1, 64, 64),
            )
            return sum([float(batched[i, i].flatten()) for i in range(8)])

        return 0.0

    def recompute_measures(self, layer_name, measures):
        with open("over_rho.json", "r") as f:
            over_rho = json.load(f)
        with open("models_values.json", "r") as f:
            models_values = json.load(f)
        per_sample_values = torch.zeros(
            (len(BIASES), len(ITERATIONS), self.len_x, len(measures) + 1)
        )
        indices = np.round(np.linspace(0, MAX_INDEX, self.len_x)).astype(int)

        # if they are not needed, initialize empty
        hm_0 = torch.zeros(8, 1, 64, 64)
        hm_1 = torch.zeros(8, 1, 64, 64)
        rel_0 = torch.zeros(8)
        rel_1 = torch.zeros(8)
        for rho_ind, rho in enumerate((pbar := tqdm(BIASES))):
            if str(rho) not in over_rho:
                over_rho[str(rho)] = {meas: 0.0 for meas in measures}
                models_values[str(rho)] = {}
            for m in ITERATIONS:
                if str(m) not in models_values:
                    models_values[str(rho)][str(m)] = {meas: 0.0 for meas in measures}
                model_name = to_name(rho, m)
                model = load_model(NAME, rho, m)
                crpa = CRPAttribution(model, self.ds, NAME, model_name)
                pbar.set_postfix(m=m)
                for ind, x in enumerate(indices):
                    latents, wm, offset = self.ds.get_item_info(x)
                    mask = torch.zeros(64, 64).to(self.tdev)
                    mask[
                        max(0, 57 + offset[0]) : max(0, 58 + offset[0]) + 5,
                        max(offset[1] + 3, 0) : max(offset[1] + 4, 0) + 10,
                    ] = 1
                    label = latents[0]
                    # image W=0, W=1
                    image_1 = self.ds.load_image_wm(x, True)
                    image_0 = self.ds.load_image_wm(x, False)
                    # if "m2_relative_mask" not in measures and len(measures) > 1:
                    # prediction output W=0, W=1
                    with torch.no_grad():
                        predv_0 = model(image_0)
                        predv_1 = model(image_1)
                    # classification W=0, W=1
                    predi_0 = int(predv_0.data.max(1)[1][0])
                    predi_1 = int(predv_1.data.max(1)[1][0])

                    per_sample_values[rho_ind, m, ind, len(ALL_MEASURES)] = x
                    # heatmaps W=0, W=1
                    if np.any(np.array([bool("m2" in meas) for meas in measures])):
                        hm_0 = self.heatmaps(image_0, mask, layer_name, crpa)
                        hm_1 = self.heatmaps(image_1, mask, layer_name, crpa)

                        # relevances W=0, W=1
                        rel_0 = self.relevances(image_0, label, layer_name, crpa)
                        rel_1 = self.relevances(image_1, label, layer_name, crpa)

                    inputvals = [
                        predv_0,
                        predv_1,
                        predi_0,
                        predi_1,
                        rel_0,
                        rel_1,
                        hm_0,
                        hm_1,
                        x,
                        label,
                        image_0,
                        image_1,
                        mask,
                        layer_name,
                        crpa,
                    ]
                    for meas_ind, meas in enumerate(measures):
                        if meas not in ["m0_mi", "m0_rho", "m0_phi"]:
                            if meas not in models_values[str(rho)][str(m)] or isnan(
                                models_values[str(rho)][str(m)][meas]
                            ):
                                models_values[str(rho)][str(m)][meas] = 0.0
                            if meas not in over_rho[str(rho)] or isnan(
                                over_rho[str(rho)][meas]
                            ):
                                over_rho[str(rho)][meas] = 0.0
                            val = self.measure_function(meas, inputvals)
                            per_sample_values[rho_ind, m, ind, meas_ind] = val
                            models_values[str(rho)][str(m)][meas] += val / self.len_x
            for m in ITERATIONS:
                for k, v in models_values[str(rho)][str(m)].items():
                    if k in measures and k not in ["m0_mi", "m0_rho", "m0_phi"]:
                        over_rho[str(rho)][k] += float(v / len(ITERATIONS))
            if "m0_phi" in measures:
                indices_long = np.round(np.linspace(0, MAX_INDEX, 40000)).astype(int)
                ds = BiasedNoisyDataset(rho)
                labels_true, labels_pred = [], []
                for ix in indices_long:
                    lats, wm, offset = ds.get_item_info(ix)
                    labels_true.append(lats[0])
                    labels_pred.append(int(wm))
                over_rho[str(rho)]["m0_mi"] = normalized_mutual_info_score(
                    labels_true, labels_pred
                )
                over_rho[str(rho)]["m0_phi"] = matthews_corrcoef(
                    labels_true, labels_pred
                )
                over_rho[str(rho)]["m0_rho"] = rho
                for m in ITERATIONS:
                    models_values[str(rho)][str(m)]["m0_mi"] = over_rho[str(rho)][
                        "m0_mi"
                    ]
                    models_values[str(rho)][str(m)]["m0_phi"] = over_rho[str(rho)][
                        "m0_phi"
                    ]
                    models_values[str(rho)][str(m)]["m0_rho"] = rho
            per_sample_values[rho_ind, :, :, len(ALL_MEASURES) - 3] = over_rho[
                str(rho)
            ]["m0_rho"]
            per_sample_values[rho_ind, :, :, len(ALL_MEASURES) - 2] = over_rho[
                str(rho)
            ]["m0_mi"]
            per_sample_values[rho_ind, :, :, len(ALL_MEASURES) - 1] = over_rho[
                str(rho)
            ]["m0_phi"]
            with open("m2_mac.pickle", "wb") as f:
                pickle.dump(per_sample_values, f, protocol=pickle.HIGHEST_PROTOCOL)

            """ with open("over_rho.json", "w") as f:
                json.dump(over_rho, f)
            with open("models_values.json", "w") as f:
                json.dump(models_values, f) """

    def recompute_gt(self):
        m1_mi = torch.zeros((len(BIASES), len(ITERATIONS), 2), dtype=torch.float)
        indices = list(np.round(np.linspace(0, MAX_INDEX, self.len_x)).astype(int))
        for rho_ind, rho in enumerate((pbar := tqdm(BIASES))):
            with torch.no_grad():
                for m in ITERATIONS:
                    unbds = BiasedNoisyDataset()
                    my_subset = Subset(unbds, indices)
                    unbiased_loader = DataLoader(my_subset, batch_size=128)
                    model = load_model(NAME, rho, m)
                    pbar.set_postfix(m=m)
                    labels_pred = []
                    labels_true = []
                    for data in unbiased_loader:
                        images, _, watermarks = data
                        pred = model(images)
                        predi = pred.data.max(1)[1].int()
                        labels_true.append(predi)
                        labels_pred.append(watermarks.long())
                    labels_true = torch.cat(labels_true)
                    labels_pred = torch.cat(labels_pred)
                    m1_mi[rho_ind, m, 0] = normalized_mutual_info_score(
                        labels_true, labels_pred
                    )
                    m1_mi[rho_ind, m, 1] = matthews_corrcoef(labels_true, labels_pred)
        with open("m1_mi_6400.pickle", "wb") as f:
            pickle.dump(m1_mi, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser("script_parallel")
    # parser.add_argument("layername", help="layer float", type=str)
    allm = AllMeasures("../dsprites-dataset/images/", 8)
    # allm.recompute_gt()
    allm.recompute_measures(
        "convolutional_layers.6",
        [
            "m2_mac_absolute",
            "m2_mac_absolute_weigh",
            "m2_mac_euclid",
            "m2_mac_euclid_weigh_norm",
            "m2_mac_euclid_weigh",
        ],
    )
""" "m2_mac_absolute",
            "m2_mac_absolute_weigh",
            "m2_mac_euclid",
            "m2_mac_euclid_weigh_norm",
            "m2_mac_euclid_weigh", """

            #"m2_mac_absolute",
            #"m2_mac_absolute_weigh",
            "m2_mac_euclid_weigh_abs_norm",
            #"m2_mac_euclid_weigh_norm",
            "m2_mac_euclid_weigh",