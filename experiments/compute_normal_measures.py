import pickle
import numpy as np
import torch
from tqdm import tqdm
import json
from os import makedirs
from os.path import isdir, isfile
import gzip
from sklearn.metrics import matthews_corrcoef, normalized_mutual_info_score
from torch.utils.data import DataLoader, Subset
from wdsprites_dataset import BiasedNoisyDataset, BackgroundDataset
from crp_attribution import CRPAttribution
from network import load_model
from helper import init_experiment, to_name
from test_dataset import TestDataset


MAX_INDEX = 491519  # 491520 is true size, but not including last
SAMPLE_SET_SIZE = 128
SEED = 42
LAYER_ID_MAP = {
    "convolutional_layers.0": 8,
    "convolutional_layers.3": 8,
    "convolutional_layers.6": 8,
    "linear_layers.0": 6,
    "linear_layers.2": 2,
}
NAME = "clustermodels/final"  # "clustermodels/model_seeded"  #
BIASES = list(np.round(np.linspace(0, 1, 51), 3))
HEATMAP_FOLDER = "random_output"
IMG_PATH = "dsprites-dataset/images/"


def find_else(element, list_element):
    try:
        index_element = list_element.index(element)
        return index_element
    except ValueError:
        return None


class PerSampleInfo:
    def __init__(self, rho, m, len_x, experiment_name=HEATMAP_FOLDER):
        self.imgs = torch.zeros((len_x, 2, 8, 64, 64), dtype=torch.float16)
        self.pred = torch.zeros((len_x, 2, 2), dtype=torch.float16)
        self.pred_int = torch.zeros((len_x, 2), dtype=torch.uint8)
        self.rel = torch.zeros((len_x, 2, 8), dtype=torch.float16)
        self.name = f"outputs/{experiment_name}/{rho}_{m}.gz"

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
        img_path=IMG_PATH,
        sample_set_size=SAMPLE_SET_SIZE,
        layer_name="convolutional_layers.6",
        model_path=NAME,
        experiment_name=HEATMAP_FOLDER,
    ) -> None:
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tdev = torch.device(self.device)
        self.max_index = MAX_INDEX
        self.len_x = sample_set_size
        self.iterations = list(range(16))

        if model_path.endswith("final"):
            self.ds = BiasedNoisyDataset(0, 0.5, False, img_path=img_path)
            self.model_type = "watermark"
            self.test_data = TestDataset(length=300, experiment=self.model_type)
        else:
            self.ds = BackgroundDataset(0, 0.5, False, img_path=img_path)
            self.model_type = "pattern"
            self.test_data = TestDataset(length=300, experiment=self.model_type)
        self.layer_name = layer_name
        self.len_neurons = LAYER_ID_MAP[self.layer_name]
        self.model_path = model_path
        self.experiment_name = experiment_name

    def heatmaps(self, image: torch.Tensor, crpa: CRPAttribution, pred: int):
        conditions = [
            {self.layer_name: [i], "y": [pred]} for i in range(self.len_neurons)
        ]
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

    def relevances(self, image, crpa: CRPAttribution, pred: int):
        attr = crpa.attribution(
            image,
            [{"y": [pred]}],
            crpa.composite,
            record_layer=crpa.layer_names,
        )
        rel_c = crpa.cc.attribute(attr.relevances[self.layer_name], abs_norm=True)
        return rel_c[0]

    def compute_per_sample(self, is_random=False):
        indices = np.round(np.linspace(0, MAX_INDEX, self.len_x)).astype(int)
        if not isdir(f"outputs/{self.experiment_name}"):
            print("creating folder", f"outputs/{self.experiment_name}")
            makedirs(f"outputs/{self.experiment_name}")

        for rho_ind, rho in enumerate((pbar := tqdm(BIASES))):
            for m in self.iterations:
                model_name = f"{self.experiment_name}_{to_name(rho, m)}"
                model = load_model(self.model_path, rho, m, self.model_type)
                crpa = CRPAttribution(
                    model, self.test_data, self.model_path, model_name
                )
                pbar.set_postfix(m=m)
                results_m_r = PerSampleInfo(
                    rho_ind, m, self.len_x, self.experiment_name
                )
                if is_random:
                    filename = f"outputs/{self.experiment_name}/{rho_ind}_{m}.gz"
                    with gzip.open(filename, mode="rb") as f:
                        r_m_info = pickle.load(f)
                    pred0s = r_m_info[1][:, 0].to(dtype=torch.float)
                    pred1s = r_m_info[1][:, 1].to(dtype=torch.float)
                    permuted = np.zeros((256, 2))
                    permuted[:128] = pred0s
                    permuted[128:] = pred1s
                    rng = np.random.default_rng(seed=SEED)
                    rng.shuffle(permuted, axis=0)
                    permuted = torch.from_numpy(permuted).view(128, 1, 2, 2)
                for ind, x in enumerate(indices):
                    # image W=0, W=1
                    image_1 = self.ds.load_image_wm(x, True)
                    image_0 = self.ds.load_image_wm(x, False)
                    # prediction output W=0, W=1
                    if is_random:
                        predv_0 = permuted[ind, :, 0]
                        predv_1 = permuted[ind, :, 1]
                    else:
                        with torch.no_grad():
                            predv_0 = model(image_0)
                            predv_1 = model(image_1)
                    # classification W=0, W=1
                    predi_0 = int(predv_0.data.max(1)[1][0])
                    predi_1 = int(predv_1.data.max(1)[1][0])
                    # heatmaps W=0, W=1
                    heatmap_0 = self.heatmaps(image_0, crpa, predi_0)
                    heatmap_1 = self.heatmaps(image_1, crpa, predi_1)
                    # relevances W=0, W=1
                    rel_0 = self.relevances(image_0, crpa, predi_0)
                    rel_1 = self.relevances(image_1, crpa, predi_1)

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
        masked = heatmaps * wm_mask
        sorted_values = torch.topk(
            heatmaps.abs().view(self.len_neurons, -1), k=mask_size + 1, dim=1
        ).values
        cutoffs = sorted_values[:, mask_size]
        cutoffs = torch.where(cutoffs > 0, cutoffs, 100)
        rrank = masked >= cutoffs[:, None, None]

        # pointing game
        largest = sorted_values[:, 0]
        non_empty = torch.count_nonzero(largest > 0)
        largest = torch.where(largest > 0.0, largest, 100.0)
        pointing_game = (
            torch.max(masked.abs().view(self.len_neurons, -1)) >= largest
        ).float()
        ## rra
        rank_counts = torch.count_nonzero(rrank, dim=(1, 2))
        rra = rank_counts / mask_size
        # rma
        rel_within = torch.sum(masked, dim=(1, 2))
        rel_total = torch.sum(heatmaps.abs(), dim=(1, 2))

        return dict(
            rel_total=rel_total,
            rel_within=rel_within,
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

    def kernel2d_l2s(self, a, b):
        return (
            torch.nn.functional.conv2d(a, a)
            - 2 * torch.nn.functional.conv2d(a, b)
            + torch.nn.functional.conv2d(b, b)
        )

    def kernel2d_cosine(self, a, b):
        return (
            1
            - (
                torch.nn.functional.conv2d(a, b)
                / ((a.abs().norm() * b.abs().norm()) + 1e-8)
            )
        ) / 2

    def kernel1d(self, a, b, dim):
        return torch.sum(torch.square(a - b), dim=dim)

    def kernel_distance(self, v0, v1, kernel="l2squared"):
        if kernel == "cosine":
            batched = self.kernel2d_cosine(
                v1.view(self.len_neurons, 1, 64, 64),
                v0.view(self.len_neurons, 1, 64, 64),
            )
        else:
            batched = self.kernel2d_l2s(
                v1.view(self.len_neurons, 1, 64, 64),
                v0.view(self.len_neurons, 1, 64, 64),
            )
        if kernel == "l2":
            return torch.sqrt(torch.abs(torch.sum(torch.diagonal(batched, 0))))
        return torch.sum(torch.diagonal(batched, 0))

    def easy_compute_measures(self):
        torch.set_grad_enabled(False)
        measures = [
            "m1_phi",
            "m1_mlc_abs",
            "m1_mlc_cosine",
            "m1_mlc_euclid",
            "m1_mlc_l2square",
            "m2_rel_abs",
            "m2_rel_cosine",
            "m2_rel_euclid",
            "m2_rel_l2square",
            "m2_mac_abs",
            "m2_mac_euclid",
            "m2_mac_cosine",
            "m2_mac_l2square",
            "m2_rma",
            "m2_rma_unweighted",
            "m2_pg_weighted",
            "m2_pg_unweighted",
            "m2_rra_weighted",
            "m2_rra_unweighted",
        ]

        def m_i(name):
            return find_else(name, measures)

        indices = np.round(np.linspace(0, MAX_INDEX, self.len_x)).astype(int)
        per_sample_values = torch.zeros(
            (len(BIASES), len(self.iterations), self.len_x, len(measures))
        )

        mostimportantneuron = torch.zeros(
            (len(BIASES), len(self.iterations), self.len_x, 3)
        )
        softmax = torch.nn.Softmax(dim=1)
        savepath = (
            f"outputs/measures/all_measures_{self.len_x}_{self.experiment_name}.pickle"
        )
        print(savepath)
        saved_until = -1
        """ saved_until_path = f"temp{saved_until}_{savepath}"
        if isfile(saved_until_path):
            with gzip.open(saved_until_path, "rb") as f:
                loaded = pickle.load(f)
            per_sample_values = loaded
            per_sample_values.requires_grad = False """
        nvals = {
            "per_image": [],
            "per_neuron_max": [],
            "per_model_max": [],
            "per_model": [],
        }
        if isfile(savepath):
            with gzip.open(savepath, "rb") as f:
                loaded = pickle.load(f)
            per_sample_values = loaded
        for rho_ind, rho in enumerate((pbar := tqdm(BIASES))):
            if rho_ind > saved_until:
                for m in self.iterations:
                    pbar.set_postfix(m=m, rho=rho)
                    filename = f"outputs/{self.experiment_name}/{rho_ind}_{m}.gz"
                    with gzip.open(filename, mode="rb") as f:
                        r_m_info = pickle.load(f)
                    hm0s = r_m_info[0][:, 0].to(dtype=torch.float)
                    hm1s = r_m_info[0][:, 1].to(dtype=torch.float)
                    pred0s = r_m_info[1][:, 0].to(dtype=torch.float)
                    pred1s = r_m_info[1][:, 1].to(dtype=torch.float)
                    rels0 = r_m_info[2][:, 0].to(dtype=torch.float)
                    rels1 = r_m_info[2][:, 1].to(dtype=torch.float)

                    # phi correlation (=prediction flip) prediction
                    """ labels_true = torch.cat((r_m_info[3][:, 0], r_m_info[3][:, 1]))
                    labels_pred = torch.cat(
                        [torch.zeros(self.len_x), torch.ones(self.len_x)]
                    )
                    per_sample_values[rho_ind, m, :, m_i("m1_phi")] =  matthews_corrcoef(
                        labels_true, labels_pred
                    ) """
                    """ per_sample_values[rho_ind, m, :, m_i("m1_phi")] = (
                        torch.count_nonzero(r_m_info[3][:, 1] != r_m_info[3][:, 0])
                        / self.len_x
                    ) """

                    # PREDICTION
                    # pred0sabs = softmax(pred0s)
                    # pred1sabs = softmax(pred1s)
                    # absolute difference predictions
                    """ per_sample_values[rho_ind, m, :, m_i("m1_mlc_abs")] = (
                        torch.sum(torch.abs(pred1sabs - pred0sabs), dim=1) / 2
                    ) """
                    # cosine distance predictions
                    # same as others, not softmaxed
                    """ per_sample_values[rho_ind, m, :, m_i("m1_mlc_cosine")] = (
                        1
                        - torch.nn.functional.cosine_similarity(
                            pred1s,
                            pred0s,
                            dim=1,
                        )
                    ) / 2 """
                    # euclid distance prediction logits
                    """ per_sample_values[rho_ind, m, :, m_i("m1_mlc_euclid")] = (
                        torch.sqrt(torch.sum(torch.square(rels1 - rels0), dim=1)) / 2
                    )
                    # kernel distance prediction logits
                    per_sample_values[rho_ind, m, :, m_i("m1_mlc_l2square")] = (
                        torch.sum(torch.square(rels1 - rels0), dim=1) / 2
                    ) """

                    """ # RELEVANCES
                    # absolute difference relevances (/len samples)
                    per_sample_values[rho_ind, m, :, m_i("m2_rel_abs")] = (
                        torch.sum(torch.abs(rels1 - rels0), dim=1) / 2
                    )
                    # euclid distance relevances
                    per_sample_values[rho_ind, m, :, m_i("m2_rel_euclid")] = torch.sqrt(
                        torch.sum(torch.square(rels1 - rels0), dim=1)
                    ) /2
                    # squared distance relevances
                    per_sample_values[rho_ind, m, :, m_i("m2_rel_l2square")] = (
                        torch.sum(torch.square((rels1 - rels0)/2), dim=1)
                    ) 
                    # cosine distance relevances
                    per_sample_values[rho_ind, m, :, m_i("m2_rel_cosine")] = (
                        self.cosine_distance(rels1, rels0) / 2
                    ) """

                    """ # HEATMAPS
                    # cosine distance heatmaps
                    per_sample_values[rho_ind, m, :, m_i("m2_mac_cosine")] = (
                        self.cosine_distance(hm1s, hm0s) / 2
                    ) """
                    """ maxval = max(
                        hm0s.abs().sum(dim=(1, 2, 3)).max(),
                        hm1s.abs().sum(dim=(1, 2, 3)).max(),
                    )
                    nvals["per_model_max"] += [float(maxval)]
                    maxval = 1 if maxval == 0 else maxval
                    hm0sabs = hm0s / maxval
                    hm1sabs = hm1s / maxval
                    # absolute difference heatmaps
                    # normalized by max total absolute relevance
                    per_sample_values[rho_ind, m, :, m_i("m2_mac_abs")] = torch.sum(
                        torch.abs(hm1sabs - hm0sabs), dim=(1, 2, 3)
                    )
                    per_sample_values[rho_ind, m, :, m_i("m2_mac_l2square")] = (
                        torch.sum(torch.square((hm1sabs - hm0sabs)), dim=(1, 2, 3))
                    ) """
                    # euclidean distance heatmaps
                    """ per_sample_values[rho_ind, m, :, m_i("m2_mac_euclid")] = torch.sqrt(
                        (torch.sum(torch.square((hm1sabs - hm0sabs)), dim=(1, 2, 3)))
                    ) """

                    if torch.any(hm1s.abs() > 0.0001) or torch.any(hm0s.abs() > 0.0001):
                        for n, i in enumerate(indices):
                            if torch.any(hm1s[n].abs() > 0.0001) or torch.any(
                                hm0s[n].abs() > 0.0001
                            ):
                                """per_sample_values[
                                    rho_ind, m, n, m_i("m2_mac_cosine")
                                ] = (self.cosine_distance(hm1s[n], hm0s[n]) / 2)
                                maxval = max(
                                    hm0s[n].abs().sum(),  # dim=(1,2)).max(
                                    hm1s[n].abs().sum(),  # dim=(1,2)).max(
                                )
                                nvals["per_image"] += [float(maxval)]
                                nvals["per_neuron_max"] += [
                                    float(
                                        max(
                                            hm0s[n].abs().sum(dim=(1, 2)).max(),
                                            hm1s[n].abs().sum(dim=(1, 2)).max(),
                                        )
                                    )
                                ]

                                if maxval > 0:
                                    hm0sabs = hm0s[n] / maxval
                                    hm1sabs = hm1s[n] / maxval
                                else:
                                    hm0sabs = hm0s[n]
                                    hm1sabs = hm1s[n]
                                # absolute difference heatmaps
                                # normalized by max total absolute relevance
                                per_sample_values[rho_ind, m, n, m_i("m2_mac_abs")] = (
                                    torch.sum(torch.abs(hm1sabs - hm0sabs))
                                )
                                per_sample_values[
                                    rho_ind, m, n, m_i("m2_mac_l2square")
                                ] = (
                                    torch.sum(torch.square((hm1sabs - hm0sabs)))
                                    * maxval
                                )"""

                                # prepare for region specific measures
                                mask = self.ds.load_watermark_mask(i)
                                mask = mask.view(64, 64).to(self.tdev)
                                inds1 = torch.topk(rels1[n], 1).indices
                                inds0 = torch.topk(rels0[n], 1).indices

                                weight = rels1[n]
                                hms_values1 = self.heatmap_values(
                                    hm1s[n], mask
                                )  # hm1sabs
                                hms_values0 = self.heatmap_values(
                                    hm0s[n], mask
                                )  # hm0sabs
                                m1 = max(
                                    float(hms_values1["rel_total"].sum()),
                                    float(hms_values0["rel_total"].sum()),
                                )
                                max_rel_tot = 1 if m1 == 0 else m1
                                nvals["per_image"] += [float(max_rel_tot)]

                                rma_1 = torch.where(
                                    hms_values1["rel_total"] > 0,
                                    hms_values1["rel_within"]
                                    / hms_values1["rel_total"],
                                    0,
                                )
                                rma_0 = torch.where(
                                    hms_values0["rel_total"] > 0,
                                    hms_values0["rel_within"]
                                    / hms_values0["rel_total"],
                                    0,
                                )

                                # rma weighted
                                per_sample_values[rho_ind, m, n, m_i("m2_rma")] = (
                                    torch.sum(torch.abs((rma_1 - rma_0) * weight))
                                )
                                # rma unweighted most important neuron
                                per_sample_values[
                                    rho_ind, m, n, m_i("m2_rma_unweighted")
                                ] = torch.sum(torch.abs(rma_1[inds1] - rma_0[inds0]))

                                """ mostimportantneuron[rho_ind, m, n, 0] = torch.sum(
                                    torch.abs(rma_1[inds1] - rma_0[inds1])
                                )

                                # pg weighted sum
                                per_sample_values[
                                    rho_ind, m, n, m_i("m2_pg_weighted")
                                ] = torch.sum(
                                    torch.abs(
                                        (hms_values1["pg"] - hms_values0["pg"]) * weight
                                    )
                                )
                                # pg unweighted most important neuron
                                per_sample_values[
                                    rho_ind, m, n, m_i("m2_pg_unweighted")
                                ] = torch.sum(
                                    torch.abs(
                                        (
                                            hms_values1["pg"][inds1]
                                            - hms_values0["pg"][inds0]
                                        )
                                    )
                                )
                                mostimportantneuron[rho_ind, m, n, 1] = torch.sum(
                                    torch.abs(
                                        (
                                            hms_values1["pg"][inds1]
                                            - hms_values0["pg"][inds1]
                                        )
                                    )
                                )
                                # rra weighted sum
                                per_sample_values[
                                    rho_ind, m, n, m_i("m2_rra_weighted")
                                ] = torch.sum(
                                    torch.abs(
                                        (hms_values1["rra"] - hms_values0["rra"])
                                        * weight
                                    )
                                )
                                # rra unweighted most important neuron
                                per_sample_values[
                                    rho_ind, m, n, m_i("m2_rra_unweighted")
                                ] = torch.sum(
                                    torch.abs(
                                        (
                                            hms_values1["rra"][inds1]
                                            - hms_values0["rra"][inds0]
                                        )
                                    )
                                )
                                mostimportantneuron[rho_ind, m, n, 2] = torch.sum(
                                    torch.abs(
                                        (
                                            hms_values1["rra"][inds1]
                                            - hms_values0["rra"][inds1]
                                        )
                                    )
                                ) """
                """ if rho_ind == 25:
                    print("save temp")
                    with gzip.open(f"temp{rho_ind}_{savepath}", "wb") as f:
                        pickle.dump(
                            per_sample_values, f, protocol=pickle.HIGHEST_PROTOCOL
                        ) """

        with gzip.open(savepath, "wb") as f:
            pickle.dump(per_sample_values, f, protocol=pickle.HIGHEST_PROTOCOL)
        """ with gzip.open(
            f"outputs/measures/mostimportantneuron_{self.experiment_name}.pickle", "w"
        ) as f:
            pickle.dump(mostimportantneuron, f, protocol=pickle.HIGHEST_PROTOCOL) """
        with open(f"outputs/measures/nvals_{self.experiment_name}.json", "w") as fj:
            json.dump(nvals, fj)

    def prediction_flip(self):
        per_sample_values = torch.zeros((len(BIASES), len(self.iterations), 1))
        for rho_ind, rho in enumerate((pbar := tqdm(BIASES))):
            for m in self.iterations:
                pbar.set_postfix(m=m)
                filename = f"outputs/{self.experiment_name}/{rho_ind}_{m}.gz"
                with gzip.open(filename, mode="rb") as f:
                    r_m_info = pickle.load(f)
                # prediction flip
                per_sample_values[rho_ind, m, 0] = (
                    torch.count_nonzero(r_m_info[3][:, 1] != r_m_info[3][:, 0])
                    / self.len_x
                )
        with open(
            f"outputs/measures/pf_{self.len_x}_{self.experiment_name}.pickle", "wb"
        ) as f:
            pickle.dump(per_sample_values, f, protocol=pickle.HIGHEST_PROTOCOL)

    def recompute_gt(self, length):
        m1_mi = torch.zeros((len(BIASES), len(self.iterations), 3), dtype=torch.float)
        indices = list(np.round(np.linspace(0, MAX_INDEX, length)).astype(int))
        my_subset = Subset(self.ds, indices)
        unbiased_loader = DataLoader(my_subset, batch_size=256)
        labels_wm = torch.tensor([int(self.ds.watermarks[ind]) for ind in indices])
        labels_true = torch.tensor([self.ds.labels[ind][1] for ind in indices])
        for rho_ind, rho in enumerate((pbar := tqdm(BIASES))):
            with torch.no_grad():
                for m in self.iterations:
                    model = load_model(self.model_path, rho, m, self.model_type)
                    pbar.set_postfix(m=m)
                    labels_pred = []
                    for data in unbiased_loader:
                        images, _ = data
                        pred = model(images)
                        predi = pred.data.max(1)[1].int()
                        labels_pred.append(predi)
                    labels_pred = torch.cat(labels_pred)

                    m1_mi[rho_ind, m, 0] = normalized_mutual_info_score(  # type: ignore
                        labels_pred, labels_wm
                    )
                    m1_mi[rho_ind, m, 1] = matthews_corrcoef(labels_pred, labels_wm)
                    # accuracy drop
                    m1_mi[rho_ind, m, 2] = (
                        torch.count_nonzero(labels_pred != labels_true) / length
                    )

        with open(
            f"outputs/measures/m1_mi_{length}_{self.experiment_name}.pickle", "wb"
        ) as f:
            pickle.dump(m1_mi, f, protocol=pickle.HIGHEST_PROTOCOL)

    def data_ground_truth(self, length):
        m0_gt = torch.zeros((len(BIASES), len(self.iterations), 4), dtype=torch.float)
        indices = list(np.round(np.linspace(0, MAX_INDEX, length)).astype(int))
        for rho_ind, rho in enumerate(tqdm(BIASES)):
            if model_path.endswith("final"):
                biased_ds = BiasedNoisyDataset(rho, 0.5, False)
            else:
                biased_ds = BackgroundDataset(rho, 0.5, False)
            labels_pred = torch.tensor(
                [int(biased_ds.watermarks[index]) for index in indices]
            )
            labels_true = torch.tensor(
                [biased_ds.labels[index][1] for index in indices]
            )
            m0_gt[rho_ind, :, 0] = rho
            m0_gt[rho_ind, :, 1] = normalized_mutual_info_score(  # type: ignore
                labels_true, labels_pred
            )
            m0_gt[rho_ind, :, 2] = matthews_corrcoef(labels_true, labels_pred)
            # accuracy drop
            m0_gt[rho_ind, :, 3] = (
                torch.count_nonzero(labels_pred != labels_true) / length
            )

        with open(
            f"outputs/measures/m0_gt_{length}_{self.experiment_name}.pickle", "wb"
        ) as f:
            pickle.dump(m0_gt, f, protocol=pickle.HIGHEST_PROTOCOL)

    def gt_shape(self, length):
        shape_gt = torch.zeros(
            (len(BIASES), len(self.iterations), 3), dtype=torch.float
        )
        indices = list(np.round(np.linspace(0, MAX_INDEX, length)).astype(int))
        for rho_ind, rho in enumerate((pbar := tqdm(BIASES))):
            with torch.no_grad():
                for m in self.iterations:
                    model = load_model(self.model_path, rho, m, self.model_type)
                    pbar.set_postfix(m=m)
                    labels_pred = []
                    labels_true = []
                    is_flipped = 0
                    for index in indices:
                        image, target = self.ds[index]
                        image_othershape = self.ds.load_flipped_latent(index)
                        image_othershape = image_othershape.view(1, 1, 64, 64)
                        image = image.view(1, 1, 64, 64)
                        pred = model(image)
                        predi = pred.data.max(1)[1].int()
                        pred_other = model(image_othershape)
                        predi_other = pred_other.data.max(1)[1].int()
                        is_flipped += abs(predi - predi_other)
                        labels_true.append(target)
                        labels_pred.append(predi)
                    labels_true = torch.tensor(labels_true)
                    labels_pred = torch.tensor(labels_pred)

                    shape_gt[rho_ind, m, 0] = normalized_mutual_info_score(  # type: ignore
                        labels_true, labels_pred
                    )
                    shape_gt[rho_ind, m, 1] = matthews_corrcoef(
                        labels_true, labels_pred
                    )
                    shape_gt[rho_ind, m, 2] = is_flipped / length
        with open(
            f"outputs/measures/shape_gt_{length}_{self.experiment_name}.pickle", "wb"
        ) as f:
            pickle.dump(shape_gt, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    (
        sample_set_size,
        _,
        layer_name,
        is_random,
        model_path,
        experiment_name,
        _,
        _,
        _,
        _,
        _,
    ) = init_experiment(1)
    allm = AllMeasures(
        sample_set_size=sample_set_size,
        layer_name=layer_name,
        model_path=model_path,
        experiment_name=experiment_name,
    )
    # allm.compute_per_sample(is_random=is_random)
    allm.easy_compute_measures()
    # allm.prediction_flip()
    # allm.data_ground_truth(6400)
    # allm.recompute_gt(6400)
    # allm.gt_shape(128)
