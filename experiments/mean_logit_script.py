import numpy as np
import json
from tqdm import tqdm

from expbasics.ground_truth_measures import GroundTruthMeasures
from expbasics.network import train_network
from expbasics.biased_noisy_dataset import get_biased_loader

EPOCHS = 3
BATCH_SIZE = 128
NAME = "../clustermodels/noise_pos"  # "models/bigm"
IMAGE_PATH = "../dsprites-dataset/images/"  # "images/"



def logit_change_evaluate(item):
    """ if "crp_ols" in item:
        return item """
    res = item
    bias = item["bias"]
    strength = item["strength"]
    learnr = item["learning_rate"]
    num_it = item["num_it"]
    train_loader = get_biased_loader(
        bias, strength, batch_size=128, verbose=False, img_path=IMAGE_PATH
    )
    model = train_network(
        train_loader,
        bias,
        strength,
        NAME,
        BATCH_SIZE,
        load=True,
        retrain=False,
        learning_rate=learnr,
        epochs=EPOCHS,
        num_it=num_it
    )

    # add change / new measure to compute here
    gm = GroundTruthMeasures(img_path=IMAGE_PATH)
    flipvalues = gm.ols_values(model)
    ols_vals = gm.ordinary_least_squares(flipvalues)
    mean_logit = gm.mean_logit_change(flipvalues)
    flip_pred = gm.ols_prediction_values(model)
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
    with open("outputs/recompute_accuracies.json", "r") as f:
        accuracies = json.load(f)
    with open("outputs/noise_pos_accuracies.json", "r") as f:
        old_accuracies = json.load(f)
    for name, item in tqdm(accuracies.items()):
        old_acc = old_accuracies[name]["train_accuracy"][2]
        new_acc = accuracies[name]["train_accuracy"][2]
        if abs(old_acc - new_acc) > 1.2:
            result = logit_change_evaluate(item)
            accuracies[name] = result

            with open("outputs/noise_pos_accuracies.json", "w") as f:
                json.dump(accuracies, f, indent=2)


if __name__ == "__main__":
    compute_with_param()
