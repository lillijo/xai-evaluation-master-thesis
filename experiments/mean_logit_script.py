import numpy as np
import json
from expbasics.ground_truth_measures import GroundTruthMeasures
from expbasics.network import train_network
from expbasics.biased_dsprites_dataset import get_biased_loader
from tqdm import tqdm

LEARNING_RATE = [0.0005, 0.001, 0.0015, 0.002]
EPOCHS = 3
STRENGTH = 0.5
BATCH_SIZE = 128
NAME = "../clustermodels/bigm"  # "models/bigm"
IMAGE_PATH = "../dsprites-dataset/images/"  # "images/"


def to_name(b, s, l):
    return "b{}-s{}-l{}".format(
        str(round(b, 2)).replace("0.", "0_"),
        str(round(s, 2)).replace("0.", "0_"),
        str(round(l, 4)).replace("0.", "0_"),
    )


def logit_change_evaluate(item):
    res = item
    bias = item["bias"]
    strength = item["strength"]
    learnr = item["learning_rate"]
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
    )
    gm = GroundTruthMeasures(img_path=IMAGE_PATH)
    # flipvalues = gm.ols_values(model)
    # ols_vals = gm.ordinary_least_squares(flipvalues)
    # mean_logit = gm.mean_logit_change(flipvalues)
    flip_pred = gm.ols_prediction_values(model)
    ols_pred = gm.ordinary_least_squares_prediction(flip_pred)
    mean_logit_pred = gm.mean_logit_change_prediction(flip_pred)

    # res["crp_ols"] = ols_vals
    # res["crp_mean_logit_change"] = mean_logit.tolist()
    res["pred_ols"] = ols_pred
    res["pred_mean_logit_change"] = mean_logit_pred.tolist()
    return res


def compute_with_param():
    with open("outputs/measures.json", "r") as f:
        accuracies = json.load(f)
    for name, item in tqdm(accuracies.items()):
        result = logit_change_evaluate(item)
        accuracies[name] = result

        with open("outputs/measures.json", "w") as f:
            json.dump(accuracies, f, indent=2)


if __name__ == "__main__":
    compute_with_param()
