from math import isnan
import pickle
import numpy as np
import torch
from tqdm import tqdm
import math
import copy
import gzip
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
ITERATIONS = list(range(16))


def find_else(element, list_element):
    try:
        index_element = list_element.index(element)
        return index_element
    except ValueError:
        return None


class PerSampleInfo:
    def __init__(self, rho, m, len_x):
        self.imgs = torch.zeros((len_x, 2, 8, 64, 64), dtype=torch.float16)
        self.pred = torch.zeros((len_x, 2, 2), dtype=torch.float16)
        self.pred_int = torch.zeros((len_x, 2), dtype=torch.uint8)
        self.rel = torch.zeros((len_x, 2, 8), dtype=torch.float16)
        self.name = f"outputs/measures/{rho}_{m}.gz"

    def add_x(self, index, hm0, hm1, pred0, pred1, predi_0, predi_1, rel0, rel1):
        self.imgs[index, 0] = hm0
        self.imgs[index, 1] = hm1
        self.pred[index, 0] = pred0[0]
        self.pred[index, 1] = pred1[0]
        self.rel[index, 0] = rel0
        self.rel[index, 1] = rel1
        self.pred_int[index, 0] = predi_0
        self.pred_int[index, 1] = predi_1

    def save(self):
        data = [self.imgs, self.pred, self.rel, self.pred_int]
        with gzip.open(self.name, "wb") as f:
            pickle.dump(data, f)


class AllMeasures:
    def __init__(
        self,
        img_path="",
        sample_set_size=SAMPLE_SET_SIZE,
        layer_name="convolutional_layers.6",
    ) -> None:
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tdev = torch.device(self.device)
        self.max_index = MAX_INDEX
        self.len_x = sample_set_size
        self.ds = BiasedNoisyDataset(0, 0.5, False, img_path=img_path)
        self.rng = np.random.default_rng()  # seed=SEED
        self.layer_name = layer_name
        self.len_neurons = LAYER_ID_MAP[self.layer_name]

    def heatmaps(
        self,
        image: torch.Tensor,
        crpa: CRPAttribution,
    ):
        conditions = [{self.layer_name: [i], "y": [1]} for i in range(self.len_neurons)]
        heatmaps = torch.zeros((self.len_neurons, 64, 64))
        for attr in crpa.attribution.generate(
            image,
            conditions,
            crpa.composite,
            record_layer=crpa.layer_names,
            verbose=False,
            batch_size=self.len_neurons,
        ):
            heatmaps = attr.heatmap
        return heatmaps

    def relevances(self, image, crpa: CRPAttribution):
        attr = crpa.attribution(
            image,
            [{"y": [1]}],
            crpa.composite,
            record_layer=crpa.layer_names,
        )
        rel_c = crpa.cc.attribute(attr.relevances[self.layer_name], abs_norm=True)
        return rel_c[0]

    def compute_per_sample(self):
        indices = np.round(np.linspace(0, MAX_INDEX, self.len_x)).astype(int)
        for rho_ind, rho in enumerate((pbar := tqdm(BIASES))):
            for m in ITERATIONS:
                model_name = to_name(rho, m)
                model = load_model(NAME, rho, m)
                crpa = CRPAttribution(model, self.ds, NAME, model_name)
                pbar.set_postfix(m=m)
                pred_0s, pred_1s = [], []
                results_m_r = PerSampleInfo(rho_ind, m, self.len_x)
                for ind, x in enumerate(indices):
                    _, _, offset = self.ds.get_item_info(x)
                    mask = torch.zeros(64, 64).to(self.tdev)
                    mask[
                        max(0, 57 + offset[0]) : max(0, 58 + offset[0]) + 5,
                        max(offset[1] + 3, 0) : max(offset[1] + 4, 0) + 10,
                    ] = 1
                    # image W=0, W=1
                    image_1 = self.ds.load_image_wm(x, True)
                    image_0 = self.ds.load_image_wm(x, False)
                    # prediction output W=0, W=1
                    with torch.no_grad():
                        predv_0 = model(image_0)
                        predv_1 = model(image_1)
                    # classification W=0, W=1
                    predi_0 = int(predv_0.data.max(1)[1][0])
                    predi_1 = int(predv_1.data.max(1)[1][0])
                    # append to lists
                    pred_0s.append(predi_0)
                    pred_1s.append(predi_1)

                    # heatmaps W=0, W=1
                    heatmap_0 = self.heatmaps(image_0, crpa)
                    heatmap_1 = self.heatmaps(image_1, crpa)
                    # relevances W=0, W=1
                    rel_0 = self.relevances(image_0, crpa)
                    rel_1 = self.relevances(image_1, crpa)

                    results_m_r.add_x(
                        ind,
                        heatmap_0,
                        heatmap_1,
                        predv_0,
                        predv_1,
                        predi_0,
                        predi_1,
                        rel_0,
                        rel_1,
                    )
                results_m_r.save()

    def heatmap_values(
        self,
        heatmaps: torch.Tensor,
        wm_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mask_size = int(wm_mask.sum())
        non_empty = torch.Tensor(0)
        heatmaps_abs = heatmaps.abs()
        masked = heatmaps_abs * wm_mask
        sorted_values = torch.sort(
            heatmaps_abs.view(self.len_neurons, -1), dim=1, descending=True
        ).values
        cutoffs = sorted_values[:, mask_size]
        cutoffs = torch.where(cutoffs > 0, cutoffs, 100)
        rrank = masked >= cutoffs[:, None, None]

        # pointing game
        largest = sorted_values[:, 0]
        non_empty = torch.count_nonzero(largest > 0)
        largest = torch.where(largest > 0, largest, 100)
        pointing_game = (
            torch.max(masked.view(self.len_neurons, -1)) >= largest
        ).float()
        ## rra
        rank_counts = torch.count_nonzero(rrank, dim=(1, 2))
        rra = rank_counts / mask_size
        # rma
        rel_within = torch.sum(masked, dim=(1, 2))
        rel_total = torch.sum(heatmaps_abs, dim=(1, 2))
        rma = rel_within / (rel_total + 1e-10)

        return dict(
            rel_total=rel_total,
            rel_within=rel_within,
            rma=rma,
            rra=rra,
            pg=pointing_game,
            non_empty=non_empty,
        )

    def cosine_distance(self, h0, h1):
        return float(
            1
            - torch.nn.functional.cosine_similarity(
                torch.flatten(h1), torch.flatten(h0), dim=0, eps=1e-10
            )
        )

    def kernel2d(self, a, b):
        return (
            torch.nn.functional.conv2d(a, a)
            - 2 * torch.nn.functional.conv2d(a, b)
            + torch.nn.functional.conv2d(b, b)
        )

    def kernel1d(self, a, b, dim):
        return torch.sum(
            torch.square(a) - 2 * (a * b) + torch.square(b),
            dim=dim,
        )

    def kernel_distance(self, v0, v1, weight=1):
        batched = self.kernel2d(
            v1.view(self.len_neurons, 1, 64, 64),
            v0.view(self.len_neurons, 1, 64, 64),
        )
        return torch.sum(torch.diagonal(batched, 0) * weight)

    def easy_compute_measures(self):
        measures = [
            "m1_phi",
            "m1_mlc_abs",
            "m1_mlc_cosine",
            "m1_mlc_kernel",
            "m2_rel_abs",
            "m2_rel_cosine",
            "m2_rel_kernel",
            "m2_mac_abs",
            "m2_mac_euclid",
            "m2_mac_cosine",
            "m2_mac_kernel",
            "m2_rma_weighted",
            "m2_pg_weighted",
            "m2_rra_weighted",
            "m2_bbox_rel",
        ]

        def m_i(name):
            return find_else(name, measures)

        indices = np.round(np.linspace(0, MAX_INDEX, self.len_x)).astype(int)
        per_sample_values = torch.zeros(
            (len(BIASES), len(ITERATIONS), self.len_x, len(measures))
        )
        with gzip.open("all_measures_128_4.pickle", "rb") as f:
            old_values = pickle.load(f)
        per_sample_values = old_values
        softmax = torch.nn.Softmax(dim=1)
        for rho_ind, rho in enumerate((pbar := tqdm(BIASES))):
            for m in ITERATIONS:
                pbar.set_postfix(m=m)
                filename = f"outputs/measures/{rho_ind}_{m}.gz"
                with gzip.open(filename, mode="rb") as f:
                    r_m_info = pickle.load(f)
                hm0s = r_m_info[0][:, 0].to(dtype=torch.float)
                hm1s = r_m_info[0][:, 1].to(dtype=torch.float)
                pred0s = r_m_info[1][:, 0].to(dtype=torch.float)
                pred1s = r_m_info[1][:, 1].to(dtype=torch.float)
                rels0 = r_m_info[2][:, 0].to(dtype=torch.float)
                rels1 = r_m_info[2][:, 1].to(dtype=torch.float)

                # phi correlation (=prediction flip) prediction
                labels_true = torch.cat((r_m_info[3][:, 0], r_m_info[3][:, 1]))
                labels_pred = torch.cat(
                    [torch.zeros(self.len_x), torch.ones(self.len_x)]
                )
                per_sample_values[rho_ind, m, :, m_i("m1_phi")] = matthews_corrcoef(
                    labels_true, labels_pred
                )

                # PREDICTION
                pred0sabs = softmax(pred0s)
                pred1sabs = softmax(pred1s)
                # absolute difference predictions
                per_sample_values[rho_ind, m, :, m_i("m1_mlc_abs")] = (
                    torch.sum(torch.abs(pred1sabs - pred0sabs), dim=1) / 2
                )
                # cosine distance predictions
                per_sample_values[rho_ind, m, :, m_i("m1_mlc_cosine")] = (
                    1
                    - torch.nn.functional.cosine_similarity(
                        pred1sabs,
                        pred0sabs,
                        dim=1,
                    )
                )
                # kernel distance prediction logits
                per_sample_values[rho_ind, m, :, m_i("m1_mlc_kernel")] = (
                    self.kernel1d(pred1sabs, pred0sabs, dim=1) / 2
                )

                # RELEVANCES
                # absolute difference relevances (/len samples)
                per_sample_values[rho_ind, m, :, m_i("m2_rel_abs")] = (
                    torch.sum(torch.abs(rels1 - rels0), dim=1) / 2
                )
                # cosine distance relevances
                per_sample_values[rho_ind, m, :, m_i("m2_rel_cosine")] = (
                    self.cosine_distance(rels1, rels0)
                )
                # kernel distance relevances
                per_sample_values[rho_ind, m, :, m_i("m2_rel_kernel")] = self.kernel1d(
                    rels1, rels0, dim=1
                )

                # HEATMAPS
                maxval = max(
                    hm0s.abs().sum(dim=(1, 2, 3)).max(),
                    hm1s.abs().sum(dim=(1, 2, 3)).max(),
                )
                hm0sabs = hm0s / maxval
                hm1sabs = hm1s / maxval
                # absolute difference heatmaps
                # normalized by max total absolute relevance
                per_sample_values[rho_ind, m, :, m_i("m2_mac_abs")] = torch.sum(
                    torch.abs(hm1sabs - hm0sabs), dim=(1, 2, 3)
                )
                # euclidean distance heatmaps
                per_sample_values[rho_ind, m, :, m_i("m2_mac_euclid")] = torch.sum(
                    torch.sqrt(torch.sum(torch.square((hm1s - hm0s)), dim=(2, 3))),
                    dim=1,
                )
                # cosine distance
                per_sample_values[rho_ind, m, :, m_i("m2_mac_cosine")] = (
                    self.cosine_distance(hm1s, hm0s)
                )

                for n, i in enumerate(indices):
                    # kernel distance
                    per_sample_values[rho_ind, m, n, m_i("m2_mac_kernel")] = (
                        self.kernel_distance(hm1s[n], hm0s[n])
                    )

                    _, _, offset = self.ds.get_item_info(i)
                    mask = torch.zeros(64, 64).to(self.tdev)
                    mask[
                        max(0, 57 + offset[0]) : max(0, 58 + offset[0]) + 5,
                        max(offset[1] + 3, 0) : max(offset[1] + 4, 0) + 10,
                    ] = 1
                    weight = rels1[n]
                    hms_values1 = self.heatmap_values(hm1s[n], mask)
                    hms_values0 = self.heatmap_values(hm0s[n], mask)
                    # rma weighted sum
                    per_sample_values[rho_ind, m, n, m_i("m2_rma_weighted")] = (
                        torch.sum(
                            torch.abs(
                                (hms_values1["rma"] - hms_values0["rma"]) * weight
                            )
                        )
                    )
                    # pg weighted sum
                    per_sample_values[rho_ind, m, n, m_i("m2_pg_weighted")] = torch.sum(
                        torch.abs((hms_values1["pg"] - hms_values0["pg"]) * weight)
                    )
                    # rra weighted sum
                    per_sample_values[rho_ind, m, n, m_i("m2_rra_weighted")] = (
                        torch.sum(
                            torch.abs(
                                (hms_values1["rra"] - hms_values0["rra"]) * weight
                            )
                        )
                    )
                    # relevance within summed
                    per_sample_values[rho_ind, m, n, m_i("m2_bbox_rel")] = torch.sum(
                        torch.abs(hms_values1["rel_within"] - hms_values0["rel_within"])
                    ) / max(
                        1, int(hms_values1["non_empty"]), int(hms_values0["non_empty"])
                    )

        with gzip.open("all_measures_128_4.pickle", "wb") as f:
            pickle.dump(per_sample_values, f, protocol=pickle.HIGHEST_PROTOCOL)

    def prediction_flip(
        self,
    ):
        per_sample_values = torch.zeros((len(BIASES), len(ITERATIONS), self.len_x, 1))
        for rho_ind, rho in enumerate((pbar := tqdm(BIASES))):
            for m in ITERATIONS:
                pbar.set_postfix(m=m)
                filename = f"outputs/measures/{rho_ind}_{m}.gz"
                with gzip.open(filename, mode="rb") as f:
                    r_m_info = pickle.load(f)
                # prediction flip
                per_sample_values[rho_ind, m, :, 0] = (
                    torch.count_nonzero(r_m_info[3][:, 1] != r_m_info[3][:, 0])
                    / self.len_x
                )
        with open("pf_128.pickle", "wb") as f:
            pickle.dump(per_sample_values, f, protocol=pickle.HIGHEST_PROTOCOL)

    def recompute_gt(self, length):
        m1_mi = torch.zeros((len(BIASES), len(ITERATIONS), 3), dtype=torch.float)
        indices = list(np.round(np.linspace(0, MAX_INDEX, length)).astype(int))
        unbds = BiasedNoisyDataset()
        my_subset = Subset(unbds, indices)
        unbiased_loader = DataLoader(my_subset, batch_size=128)
        for rho_ind, rho in enumerate((pbar := tqdm(BIASES))):
            with torch.no_grad():
                for m in ITERATIONS:
                    model = load_model(NAME, rho, m)
                    pbar.set_postfix(m=m)
                    labels_pred = []
                    labels_wm = []
                    labels_true = []
                    for data in unbiased_loader:
                        images, targets, watermarks = data
                        pred = model(images)
                        predi = pred.data.max(1)[1].int()
                        labels_true.append(targets)
                        labels_pred.append(predi)
                        labels_wm.append(watermarks.long())
                    labels_pred = torch.cat(labels_pred)
                    labels_true = torch.cat(labels_true)
                    labels_wm = torch.cat(labels_wm)

                    m1_mi[rho_ind, m, 0] = normalized_mutual_info_score(  # type: ignore
                        labels_pred, labels_wm
                    )
                    m1_mi[rho_ind, m, 1] = matthews_corrcoef(labels_pred, labels_wm)
                    # accuracy drop
                    m1_mi[rho_ind, m, 2] = (
                        torch.count_nonzero(labels_pred != labels_true) / length
                    )

        with open(f"m1_mi_{length}.pickle", "wb") as f:
            pickle.dump(m1_mi, f, protocol=pickle.HIGHEST_PROTOCOL)

    def gt_shape(self, length):
        shape_gt = torch.zeros((len(BIASES), len(ITERATIONS), 2), dtype=torch.float)
        indices = list(np.round(np.linspace(0, MAX_INDEX, length)).astype(int))
        unbds = BiasedNoisyDataset()
        my_subset = Subset(unbds, indices)
        unbiased_loader = DataLoader(my_subset, batch_size=128)
        for rho_ind, rho in enumerate((pbar := tqdm(BIASES))):
            with torch.no_grad():
                for m in ITERATIONS:
                    model = load_model(NAME, rho, m)
                    pbar.set_postfix(m=m)
                    labels_pred = []
                    labels_true = []
                    for data in unbiased_loader:
                        images, targets, _ = data
                        pred = model(images)
                        predi = pred.data.max(1)[1].int()
                        labels_true.append(targets)
                        labels_pred.append(predi)
                    labels_true = torch.cat(labels_true)
                    labels_pred = torch.cat(labels_pred)

                    shape_gt[rho_ind, m, 0] = normalized_mutual_info_score(  # type: ignore
                        labels_true, labels_pred
                    )
                    shape_gt[rho_ind, m, 1] = matthews_corrcoef(
                        labels_true, labels_pred
                    )
        with open("shape_gt_6400.pickle", "wb") as f:
            pickle.dump(shape_gt, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser("script_parallel")
    # parser.add_argument("layername", help="layer float", type=str)
    allm = AllMeasures("../dsprites-dataset/images/", 128, "convolutional_layers.6")
    # allm.compute_per_sample()
    allm.easy_compute_measures()
    # allm.recompute_gt(128)
    # allm.prediction_flip()
    # allm.gt_shape(6400)
