import pickle
import numpy as np
import torch
from tqdm import tqdm
import json
import copy
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
        max_target="sum",
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
        self.max_target = max_target

    def relevances(self, image, crpa: CRPAttribution, pred: int):
        attr = crpa.attribution(
            image,
            [{"y": [pred]}],
            crpa.composite,
            record_layer=crpa.layer_names,
        )
        rel_c = crpa.cc.attribute(attr.relevances[self.layer_name], abs_norm=True)
        return rel_c[0]

    def compute_for_other_latent_factors(self, recompute_values=True):
        latent_factors = torch.zeros(len(BIASES), len(self.iterations), 3)
        indices = np.round(np.linspace(0, MAX_INDEX, self.len_x)).astype(int)
        softmax = torch.nn.Softmax(dim=1)
        for rho_ind, rho in enumerate((pbar := tqdm(BIASES))):
            for m in self.iterations:
                pbar.set_postfix(m=m, rho=rho)
                savepath = f"outputs/{experiment_name}/{rho}_{m}_shape.gz"
                if recompute_values or not isfile(savepath):
                    model_name = f"{self.experiment_name}_{to_name(rho, m)}"
                    model = load_model(self.model_path, rho, m, self.model_type)
                    crpa = CRPAttribution(
                        model, self.test_data, self.model_path, model_name
                    )
                    everything = []
                    for ind, index in enumerate(indices):
                        latents = self.ds.index_to_latent(index)
                        original_image, target = self.ds[index]
                        original_image = original_image.view(1, 1, 64, 64)
                        original_image.requires_grad = True
                        pred_original = model(original_image)
                        predindex_original = int(pred_original.data.max(1)[1][0])
                        rel_original = self.relevances(
                            original_image, crpa, predindex_original
                        )
                        everything.append(
                            [
                                1,
                                target,
                                latents,
                                predindex_original,
                                pred_original,
                                rel_original,
                                index,
                            ]
                        )
                        for lat in range(1, 6):
                            if lat == 1:
                                other_latent = (latents[lat] + 1) % 2
                            else:
                                other_latent = np.random.choice(
                                    [
                                        a
                                        for a in range(self.ds.latents_sizes[lat])
                                        if a != latents[lat]
                                    ]
                                )
                            other_image, other_latents = (
                                self.ds.load_flipped_each_latent(
                                    index, lat, other_latent
                                )
                            )
                            pred_flipped = model(other_image)
                            predindex_flipped = int(pred_flipped.data.max(1)[1][0])
                            rel_flipped = self.relevances(
                                other_image, crpa, predindex_flipped
                            )
                            everything.append(
                                [
                                    lat,
                                    other_latent,
                                    other_latents,
                                    predindex_flipped,
                                    pred_flipped,
                                    rel_flipped,
                                    index,
                                ]
                            )
                else:
                    with gzip.open(savepath, mode="rb") as f:
                        r_m_info = pickle.load(f)
                        everything = r_m_info["everything"]
                        latent_factors[rho_ind, m] = r_m_info["latent_factors"]
        
                original_pred = np.array([a[3] for a in everything if a[0] == 1])
                original_output = np.array(
                    [softmax(a[4]).detach().numpy() for a in everything if a[0] == 1]
                )
                original_relevance = np.array(
                    [a[5].detach().numpy() for a in everything if a[0] == 1]
                )
                for lat in range(1, 6):
                    cond = lambda a: a[1] == 0 if lat == 1 else lambda a: a[0] == lat
                    other_pred = np.array([a[3] for a in everything if cond(a)])
                    other_output = np.array(
                        [softmax(a[4]).detach().numpy() for a in everything if cond(a)]
                    )
                    other_relevance = np.array(
                        [a[5].detach().numpy() for a in everything if cond(a)]
                    )

                    prediction_flip = (
                        np.sum(np.abs(original_pred - other_pred)) / self.len_x
                    )
                    mean_logit_change = np.sum(
                        np.abs(original_output - other_output)
                    ) / (2 * self.len_x)
                    mean_relevance_change = np.sum(
                        np.abs(original_relevance - other_relevance)
                    ) / (8 * self.len_x)

                    latent_factors[rho_ind, m, (lat - 1) * 3] = prediction_flip
                    latent_factors[rho_ind, m, (lat - 1) * 3 + 1] = mean_logit_change
                    latent_factors[rho_ind, m, (lat - 1) * 3 + 2] = (
                        mean_relevance_change
                    )
                to_save_vals = dict(
                    everything=everything, latent_factors=latent_factors[rho_ind, m]
                )
                with gzip.open(savepath, "wb") as f:
                    pickle.dump(to_save_vals, f, protocol=pickle.HIGHEST_PROTOCOL)
            allmeasures_path = (
                f"latent_factors_shape_{self.len_x}_{self.experiment_name}.pickle"
            )
            with gzip.open(allmeasures_path, "wb") as f:
                pickle.dump(latent_factors, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    # Experiment 1:
    """model_path = "../clustermodels/final"
    experiment_name = "attribution_output"
    sample_set_size = 128
    layer_name = "convolutional_layers.6"
    is_random = False"""
    # model_type = "watermark"
    # iterations = 16
    # datasettype = BiasedNoisyDataset
    # mask = "bounding_box"
    # accuracypath = "outputs/retrain.json"
    # relsetds = TestDataset(length=300, im_dir="watermark_test_data")

    # Experiment 2:
    model_path = "../clustermodels/background"
    experiment_name = "overlap_attribution"
    sample_set_size = 128
    layer_name = "convolutional_layers.6"
    is_random = False
    # model_type = "overlap"
    # iterations = 10
    # datasettype = BackgroundDataset
    # mask = "shape"
    # accuracypath = "outputs/overlap1.json"
    # relsetds = TestDatasetBackground(length=300, im_dir="overlap_test_data")
    allm = AllMeasures(
        sample_set_size=sample_set_size,
        layer_name=layer_name,
        model_path=model_path,
        experiment_name=experiment_name,
    )
    # allm.compute_relevance_maximization()
    allm.compute_for_other_latent_factors(recompute_values=False)
