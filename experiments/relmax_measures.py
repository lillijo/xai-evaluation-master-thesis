import pickle
import numpy as np
import torch
from tqdm import tqdm
import math
from time import sleep
from os import makedirs
from os.path import isdir, isfile
import gzip
from sklearn.metrics import matthews_corrcoef, normalized_mutual_info_score
from torch.utils.data import DataLoader, Subset
from expbasics.biased_noisy_dataset import BiasedNoisyDataset
from expbasics.crp_attribution import CRPAttribution, get_bbox
from expbasics.network import load_model
from expbasics.background_dataset import BackgroundDataset
from expbasics.test_dataset import TestDataset
from expbasics.test_dataset_background import TestDataset as TestDatasetBackground


def to_name(b, i):
    return "b{}-i{}".format(
        str(round(b, 2)).replace(".", "_"),
        str(i),
    )


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
NAME = "../clustermodels/final"  # "../clustermodels/model_seeded"  #
BIASES = list(np.round(np.linspace(0, 1, 51), 3))
HEATMAP_FOLDER = "random_output"
IMG_PATH = "../dsprites-dataset/images/"


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
        if model_path.endswith("final"):
            self.ds = BiasedNoisyDataset(0, 0.5, False, img_path=img_path)
            self.iterations = list(range(16))
            self.model_type = "watermark"
            self.test_data = TestDataset(length=300, im_dir="watermark_test_data")
        else:
            self.ds = BackgroundDataset(0, 0.5, False, img_path=img_path)
            self.iterations = list(range(10))
            self.model_type = "overlap"
            self.test_data = TestDatasetBackground(
                length=300, im_dir="overlap_test_data"
            )
        self.layer_name = layer_name
        self.len_neurons = LAYER_ID_MAP[self.layer_name]
        self.model_path = model_path
        self.experiment_name = experiment_name

    def compute_relevance_maximization(self):
        rel_max_measures = torch.zeros(len(BIASES), len(self.iterations), 3)
        savepath = f"relmax_measures_{self.len_x}_{self.experiment_name}.pickle"
        for rho_ind, rho in enumerate(BIASES):
            for m in self.iterations:
                model_name = f"{self.experiment_name}_{to_name(rho, m)}"
                model = load_model(self.model_path, rho, m, self.model_type)
                crpa = CRPAttribution(
                    model, self.test_data, self.model_path, model_name
                )
                crpa.compute_feature_vis()
                sleep(1)
                # print("start measures", rho, m)
                len_set = 8
                filename = f"outputs/{self.experiment_name}/{rho_ind}_{m}.gz"
                with gzip.open(filename, mode="rb") as f:
                    r_m_info = pickle.load(f)
                rels0 = r_m_info[2][:, 0].to(dtype=torch.float)
                rels1 = r_m_info[2][:, 1].to(dtype=torch.float)
                with open(
                    f"{crpa.fv_path}/RelMax_sum_normed/convolutional_layers.6_data.npy",
                    "rb",
                ) as f:
                    data = torch.from_numpy(np.load(f))
                ref_c = crpa.fv.get_max_reference(
                    list(range(self.len_neurons)),
                    self.layer_name,
                    "relevance",
                    (0, len_set),
                    composite=crpa.composite,
                    rf=True,
                    plot_fn=get_bbox,
                )
                info = torch.zeros((self.len_neurons, len_set, 3))
                overlap = torch.zeros((self.len_neurons, 2))
                for neuron in range(self.len_neurons):
                    indices = data[:len_set, neuron]
                    for i, ind in enumerate(indices):
                        ind = int(ind)
                        img, label = self.test_data[ind]
                        (latents, has_watermark, offset) = self.test_data.get_item_info(
                            ind
                        )
                        info[neuron, i, 0] = label
                        info[neuron, i, 1] = int(has_watermark)
                        if has_watermark:
                            m1 = self.test_data.load_watermark_mask(ind)
                            m2 = torch.zeros(64, 64)
                            m2[
                                ref_c[neuron][i][0] : ref_c[neuron][i][1],
                                ref_c[neuron][i][2] : ref_c[neuron][i][3],
                            ] = 1
                            jaccard = (m1 * m2).sum() / (
                                m1.sum() + m2.sum() - (m1 * m2).sum()
                            )
                            info[neuron, i, 2] = jaccard
                    overlap[neuron, 0] = torch.count_nonzero(info[neuron, :, 1]) / (
                        max(
                            torch.count_nonzero(info[neuron, :, 0] == 1),
                            torch.count_nonzero(info[neuron, :, 0] == 0),
                        )
                    )
                    overlap[neuron, 1] = torch.sum(info[neuron, :, 2])
                rel_max_measures[rho_ind, m, 0] = torch.sum(
                    torch.abs(
                        rels1.mean(dim=0) * overlap[:, 0]
                        - rels0.mean(dim=0) * overlap[:, 0]
                    )
                )
                rel_max_measures[rho_ind, m, 1] = torch.sum(
                    torch.abs((overlap[:, 1]) * rels1.mean(dim=0))
                )
                rel_max_measures[rho_ind, m, 2] = torch.sum(
                    torch.abs((overlap[:, 0]) * rels1.mean(dim=0))
                )

        with gzip.open(savepath, "wb") as f:
            pickle.dump(rel_max_measures, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser("script_parallel")
    # parser.add_argument("layername", help="layer float", type=str)
    model_path = "../clustermodels/background"
    experiment_name = "overlap_attribution"
    sample_set_size = 128
    layer_name = "convolutional_layers.6"
    is_random = False
    model_type = "overlap"
    iterations = 10
    datasettype = BackgroundDataset
    mask = "shape"
    accuracypath = "outputs/overlap1.json"
    relsetds = TestDatasetBackground(length=300, im_dir="overlap_test_data")
    allm = AllMeasures(
        sample_set_size=sample_set_size,
        layer_name=layer_name,
        model_path=model_path,
        experiment_name=experiment_name,
    )
    allm.compute_relevance_maximization()
