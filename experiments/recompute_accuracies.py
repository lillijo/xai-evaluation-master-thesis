import numpy as np
import json
from tqdm import tqdm

from expbasics.network import train_network, accuracy_per_class
from expbasics.biased_noisy_dataset import get_biased_loader

EPOCHS = 3
BATCH_SIZE = 128
NAME = "../clustermodels/noise_pos"  # "../clustermodels/noise_pos"
IMAGE_PATH = "../dsprites-dataset/images/"  # "../dsprites-dataset/images/"


def recompute_accs(allwm, nowm, item):
    res = item
    bias = item["bias"]
    strength = item["strength"]
    learnr = item["learning_rate"]
    num_it = item["num_it"]
    train_loader = get_biased_loader(
        bias, strength, batch_size=128, verbose=False, split=0.3, img_path=IMAGE_PATH
    )
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
    test_loader = get_biased_loader(
        bias, strength, batch_size=128, verbose=False, split=0.01, img_path=IMAGE_PATH
    )
    res["train_accuracy"] = list(accuracy_per_class(model, test_loader))
    res["all_wm_accuracy"] = list(accuracy_per_class(model, allwm))
    res["no_wm_accuracy"] = list(accuracy_per_class(model, nowm))

    return res


def compute_with_param():
    allwm = get_biased_loader(
        0.0, 0.0, batch_size=128, verbose=False, split=0.01, img_path=IMAGE_PATH
    )
    nowm = get_biased_loader(
        0.0, 1.0, batch_size=128, verbose=False, split=0.01, img_path=IMAGE_PATH
    )
    with open("outputs/recompute_accuracies.json", "r") as f:
        accuracies = json.load(f)
        for name, item in accuracies.items():
            if not (name in accuracies and accuracies[name]["train_accuracy"][2] > 95):
                print(name)
                result = recompute_accs(allwm, nowm, item)
                accuracies[name] = result

            with open("outputs/recompute_accuracies.json", "w") as f:
                json.dump(accuracies, f, indent=2)


if __name__ == "__main__":
    compute_with_param()
