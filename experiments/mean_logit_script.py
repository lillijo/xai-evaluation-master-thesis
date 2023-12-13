import numpy as np
import json
from tqdm import tqdm

from expbasics.ground_truth_measures import GroundTruthMeasures
from expbasics.network import load_model
from expbasics.biased_noisy_dataset import get_biased_loader
from expbasics.biased_noisy_dataset import BiasedNoisyDataset

EPOCHS = 3
BATCH_SIZE = 128
NAME = "../clustermodels/noise_pos"  # "models/bigm"
IMAGE_PATH = "../dsprites-dataset/images/"  # "images/"
LAYER_NAME = "linear_layers.0"  #"convolutional_layers.6"


def logit_change_evaluate(item):
    """ if "crp_ols" in item:
        return item """
    res = item
    bias = item["bias"]
    strength = item["strength"]
    num_it = item["num_it"]
    model = load_model(NAME, bias, num_it)
    ds = BiasedNoisyDataset(bias, 0.5, img_path=IMAGE_PATH)
    # add change / new measure to compute here
    gm = GroundTruthMeasures(ds)
    flipvalues = gm.intervened_attributions(model, LAYER_NAME)
    ols_vals = gm.ordinary_least_squares(flipvalues)
    mean_logit = gm.mean_logit_change(flipvalues)
    flip_pred = gm.intervened_predictions(model)
    ols_pred = gm.ordinary_least_squares_prediction(flip_pred)
    mean_logit_pred = gm.mean_logit_change_prediction(flip_pred)
    prediction_flip = gm.prediction_flip(flip_pred).tolist()

    res["crp_ols"] = ols_vals
    res["crp_mean_logit_change"] = mean_logit.tolist()
    res["pred_ols"] = [a[0] for a in ols_pred]
    res["pred_mean_logit_change"] = mean_logit_pred.tolist()
    res["pred_flip"] = prediction_flip
    return res


def compute_with_param():
    with open("outputs/noise_pos_accuracies.json", "r") as f:
        accuracies = json.load(f)
    for name, item in tqdm(accuracies.items()):
        result = logit_change_evaluate(item)
        accuracies[name] = result

        with open("outputs/lin_ground_truth.json", "w") as f:
            json.dump(accuracies, f, indent=2)


if __name__ == "__main__":
    compute_with_param()
