import numpy as np
import json
from tqdm import tqdm

from network import train_network, accuracy_per_class
from biased_noisy_dataset import get_biased_loader

EPOCHS = 3
BATCH_SIZE = 128
NAME = "models/noise_pos"  # "../clustermodels/noise_pos"
IMAGE_PATH = "images/"  # "../dsprites-dataset/images/"


def recompute_accs(allwm, nowm, item):
    res = item
    bias = item["bias"]
    strength = item["strength"]
    learnr = item["learning_rate"]
    num_it = item["num_it"]
    test_loader = get_biased_loader(
        bias, strength, batch_size=128, verbose=False, split=0.05, img_path=IMAGE_PATH
    )
    model = train_network(
        test_loader,
        bias,
        strength,
        NAME,
        BATCH_SIZE,
        load=True,
        retrain=False,
        learning_rate=learnr,
        epochs=EPOCHS,
        num_it=num_it,
    )
    res["train_accuracy"] = list(accuracy_per_class(model, test_loader))
    print(res["train_accuracy"])
    res["all_wm_accuracy"] = list(accuracy_per_class(model, allwm))
    res["no_wm_accuracy"] = list(accuracy_per_class(model, nowm))

    return res


def compute_with_param():
    allwm = get_biased_loader(
        0.0, 0.0, batch_size=128, verbose=False, split=0.05, img_path=IMAGE_PATH
    )
    nowm = get_biased_loader(
        0.0, 1.0, batch_size=128, verbose=False, split=0.05, img_path=IMAGE_PATH
    )
    with open("parallel_accuracies.json", "r") as f:
        accuracies = json.load(f)
        for name, item in tqdm(accuracies.items()):
            result = recompute_accs(allwm, nowm, item)
            accuracies[name] = result

        with open("recompute_accuracies.json", "w") as f:
            json.dump(accuracies, f, indent=2)


if __name__ == "__main__":
    compute_with_param()
