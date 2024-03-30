import pickle
import numpy as np
import torch
import argparse
from tqdm import tqdm
import gzip
from wdsprites_dataset import BiasedNoisyDataset, BackgroundDataset
from crp_attribution import CRPAttribution
from helper import init_experiment, to_name
from network import load_model
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
        max_target="sum",
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
        self.max_target = max_target

    def load_values(self, path, len_set):
        values = {}
        for aggreg in ["max", "sum"]:
            for target in ["Rel", "Act"]:
                data = np.load(
                    f"{path}/{target}Max_{aggreg}_normed/{self.layer_name}_data.npy",
                )
                rels = np.load(
                    f"{path}/{target}Max_{aggreg}_normed/{self.layer_name}_rel.npy",
                )
                stats0 = np.load(
                    f"{path}/{target}Stats_{aggreg}_normed/{self.layer_name}/0_data.npy",
                )
                stats1 = np.load(
                    f"{path}/{target}Stats_{aggreg}_normed/{self.layer_name}/1_data.npy",
                )
                statsrel0 = np.load(
                    f"{path}/{target}Stats_{aggreg}_normed/{self.layer_name}/0_rel.npy",
                )
                statsrel1 = np.load(
                    f"{path}/{target}Stats_{aggreg}_normed/{self.layer_name}/1_rel.npy",
                )

                values[f"{target}_{aggreg}"] = {
                    "data": torch.from_numpy(
                        np.asarray(data[:len_set], dtype=np.int64)
                    ),
                    "rels": torch.from_numpy(np.asarray(rels[:len_set])),
                    "stats0": torch.from_numpy(
                        np.asarray(stats0[:len_set], dtype=np.int64)
                    ),
                    "stats1": torch.from_numpy(
                        np.asarray(stats1[:len_set], dtype=np.int64)
                    ),
                    "statsrel0": torch.from_numpy(np.asarray(statsrel0[:len_set])),
                    "statsrel1": torch.from_numpy(np.asarray(statsrel1[:len_set])),
                }

        return values

    def compute_relevance_maximization(self):
        for rho_ind, rho in enumerate(BIASES):
            for m in range(0, 16):
                model_name = f"{self.experiment_name}_{to_name(rho, m)}"
                model = load_model(self.model_path, rho, m, self.model_type)
                print(model_name, "sum")
                crpa = CRPAttribution(
                    model, self.test_data, self.model_path, model_name, max_target="sum"
                )
                crpa.compute_feature_vis()
                print(model_name, "max")
                crpa = CRPAttribution(
                    model, self.test_data, self.model_path, model_name, max_target="max"
                )
                crpa.compute_feature_vis()

    def compute_relmax_measures(self):
        rel_max_measures = torch.zeros(len(BIASES), len(self.iterations), 20)
        savepath = f"outputs/measures/relmax_measures_{self.len_x}_{self.experiment_name}.pickle"
        targets = torch.tensor([self.test_data.labels[a]["target"] for a in range(300)])
        watermarks = torch.tensor(
            [int(self.test_data.labels[a]["has_watermark"]) for a in range(300)]
        )
        for rho_ind, rho in enumerate((pbar := tqdm(BIASES))):
            for m in self.iterations:
                pbar.set_postfix(m=m, rho=rho)
                model_name = f"{self.experiment_name}_{to_name(rho, m)}"
                model = load_model(self.model_path, rho, m, self.model_type)
                crpa = CRPAttribution(
                    model, self.test_data, self.model_path, model_name
                )
                len_set = 15
                filename = f"outputs/{self.experiment_name}/{rho_ind}_{m}.gz"
                with gzip.open(filename, mode="rb") as f:
                    r_m_info = pickle.load(f)
                rels0 = r_m_info[2][:, 0].to(dtype=torch.float)
                rels1 = r_m_info[2][:, 1].to(dtype=torch.float)
                # Indices of all Maximizations:
                values = self.load_values(crpa.fv_path, len_set)
                relmax_vals = {}
                for k, maximiz in values.items():
                    overlap = torch.zeros((self.len_neurons, 10))
                    for neuron in range(self.len_neurons):
                        # measure with_wm / max(shape)
                        indices = maximiz["data"][:, neuron]
                        overlap[neuron, 0] = (
                            torch.count_nonzero(watermarks[indices]) / len_set
                        )
                        """ max(
                            float(torch.count_nonzero(targets[indices] == 1)),
                            float(torch.count_nonzero(targets[indices] == 0)),
                        ) """

                        overlap[neuron, 1] = torch.count_nonzero(
                            watermarks[indices]
                        ) / max(
                            float(torch.count_nonzero(targets[indices] == 1)),
                            float(torch.count_nonzero(targets[indices] == 0)),
                        )

                        # mean relevance of images with watermark
                        """ wm_indices = torch.where(watermarks[indices] == 1)
                        if len(wm_indices[0]) > 0:
                            rels = maximiz["rels"][wm_indices]
                            overlap[neuron, 1] = torch.mean(rels[:, neuron], dim=0) """

                        # mean relevance of images no watermark
                        nwm_indices = torch.where(watermarks[indices] == 0)
                        if len(nwm_indices[0]) > 0:
                            rels = maximiz["rels"][nwm_indices]
                            overlap[neuron, 6] = torch.mean(rels[:, neuron], dim=0)
                        overlap[neuron, 9] = maximiz["rels"][:, neuron].mean()
                        # stats relevance
                        statsindices0 = maximiz["stats0"][:, neuron]
                        statsindices1 = maximiz["stats1"][:, neuron]

                        overlap[neuron, 2] = (
                            torch.count_nonzero(watermarks[statsindices0] == 1)
                        ) / (len_set)
                        overlap[neuron, 3] = (
                            torch.count_nonzero(watermarks[statsindices1] == 1)
                        ) / (len_set)

                        # stats: shape = 0, wm = 0
                        # number with watermark
                        overlap[neuron, 4] = (
                            torch.count_nonzero(watermarks[statsindices0] == 1)
                            / len_set
                        )
                        # number without watermark
                        overlap[neuron, 7] = (
                            torch.count_nonzero(watermarks[statsindices0] == 0)
                            / len_set
                        )

                        # stats: shape = 1, wm = 1
                        # number with watermark
                        overlap[neuron, 5] = (
                            torch.count_nonzero(watermarks[statsindices1] == 1)
                            / len_set
                        )
                        # number without watermark
                        overlap[neuron, 8] = (
                            torch.count_nonzero(watermarks[statsindices1] == 0)
                            / len_set
                        )

                    inds1 = torch.topk(rels1.mean(dim=0), 1).indices
                    inds0 = torch.topk(rels0.mean(dim=0), 1).indices

                    # general reference sets
                    relmax_vals[f"{k}_diff"] = torch.abs(
                        overlap[inds1, 0] - overlap[inds0, 0]
                    )

                    relmax_vals[f"{k}_m_rels"] = (
                        torch.abs(overlap[inds1, 1] - overlap[inds0, 1]) / 2
                    )

                    # class specific reference sets
                    # for both classes averaged
                    # 3 = (s=1), 2=(s=0)
                    relmax_vals[f"{k}_stats"] = (
                        torch.abs((overlap[inds1, 3] - (overlap[inds0, 3])))
                        + torch.abs((overlap[inds1, 2] - (overlap[inds0, 2])))
                    ) / 2

                    # difference for stats images
                    relmax_vals[f"{k}_stats_diff"] = (
                        torch.abs(
                            (overlap[inds1, 5] + overlap[inds0, 7])
                            - (overlap[inds0, 8] + overlap[inds1, 4])
                        )
                        / 2
                    )

                for vi, val in enumerate(relmax_vals.values()):
                    rel_max_measures[rho_ind, m, vi] = val

        with gzip.open(savepath, "wb") as f:
            pickle.dump(rel_max_measures, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("experiment")
    parser.add_argument("experiment", help="experiment int", type=int, default=1)
    parser.add_argument("recompute", help="recompute bool", type=bool, default=False)
    args = parser.parse_args()

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
    ) = init_experiment(args.experiment)
    allm = AllMeasures(
        sample_set_size=sample_set_size,
        layer_name=layer_name,
        model_path=model_path,
        experiment_name=experiment_name,
    )
    if args.recompute:
        allm.compute_relevance_maximization()
    allm.compute_relmax_measures()

