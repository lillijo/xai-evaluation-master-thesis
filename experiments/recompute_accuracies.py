import numpy as np
import json
from tqdm import tqdm

from expbasics.network import train_network, accuracy_per_class
from expbasics.biased_noisy_dataset import get_biased_loader, BiasedNoisyDataset
from expbasics.ground_truth_measures import GroundTruthMeasures
from torch.utils.data import DataLoader, random_split

EPOCHS = 3
BATCH_SIZE = 128
NAME = "../clustermodels/model"  # "../clustermodels/noise_pos"
IMAGE_PATH = "../dsprites-dataset/images/"  # "../dsprites-dataset/images/"
#LAYER_NAME = "convolutional_layers.6"
LAYER_NAME = "linear_layers.0"


def recompute_accs(allwm, nowm, item):
    res = item
    bias = item["bias"]
    strength = item["strength"]
    learnr = item["learning_rate"]
    num_it = item["num_it"]
    ds = BiasedNoisyDataset(bias, strength, img_path=IMAGE_PATH)
    trainds, testds, _ = random_split(ds, [0.3, 0.03, 0.67])
    train_loader = DataLoader(trainds, batch_size=128, shuffle=True)
    print(bias, strength, learnr, num_it, NAME)
    model = train_network(
        train_loader,
        bias,
        strength,
        NAME,
        BATCH_SIZE,
        load=False,
        retrain=False,
        learning_rate=learnr,
        epochs=EPOCHS,
        num_it=num_it,
    )
    test_loader = DataLoader(testds, batch_size=128, shuffle=True)
    res["train_accuracy"] = list(accuracy_per_class(model, test_loader))
    res["all_wm_accuracy"] = list(accuracy_per_class(model, allwm))
    res["no_wm_accuracy"] = list(accuracy_per_class(model, nowm))

    gm = GroundTruthMeasures(ds)
    flipvalues = gm.intervened_attributions(model, LAYER_NAME)
    ols_vals = gm.ordinary_least_squares(flipvalues)
    mean_logit = gm.mean_logit_change(flipvalues)
    flip_pred = gm.intervened_predictions(model)
    ols_pred = gm.ordinary_least_squares_prediction(flip_pred)
    mean_logit_pred = gm.mean_logit_change_prediction(flip_pred)
    prediction_flip = gm.prediction_flip(flip_pred).tolist()

    res["crp_ols"] = ols_vals
    res["crp_mlc"] = mean_logit.tolist()
    res["pred_ols"] = [a[0] for a in ols_pred]
    res["pred_mlc"] = mean_logit_pred.tolist()
    res["pred_flip"] = prediction_flip

    return res


def compute_with_param():
    allwm = get_biased_loader(
        0.0, 0.0, batch_size=128, verbose=False, split=0.01, img_path=IMAGE_PATH
    )
    nowm = get_biased_loader(
        0.0, 1.0, batch_size=128, verbose=False, split=0.01, img_path=IMAGE_PATH
    )
    with open("outputs/model_accuracies.json", "r") as f:
        accuracies = json.load(f)
        for name, item in accuracies.items():
            if accuracies[name]["train_accuracy"][2] < 95 or "crp_ols" not in accuracies[name]:
                print(name)
                result = recompute_accs(allwm, nowm, item)
                accuracies[name] = result

            with open("outputs/blub_accuracies.json", "w") as f:
                json.dump(accuracies, f, indent=2)


if __name__ == "__main__":
    compute_with_param()
