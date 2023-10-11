import numpy as np
import json
from tqdm import tqdm

from crp_attribution_binary import CRPAttribution
from crp_hierarchies import sample_from_categories
from network_binary import train_network, accuracy_per_class
from biased_dsprites_dataset import get_test_dataset, get_biased_loader
from ground_truth_measures import GroundTruthMeasures

LEARNING_RATES = [
    0.001,
    0.008,
    0.1,
]  # np.round(np.linspace(0.008, 0.017, 10), 3)
BATCH_SIZE = 128
NAME = "../clustermodels/fine"
FV_NAME = "fv_model"
ITEMS_PER_CLASS = 245760
OLD_LR = 0.016


def to_name(b, s, l):
    return "b{}-s{}-l{}".format(
        str(round(b, 2)).replace("0.", "0_"),
        str(round(s, 2)).replace("0.", "0_"),
        str(round(l, 2)).replace("0.", "0_"),
    )


def train_model_evaluate(name, item, gm):
    res = item
    print(name)
    train_loader = get_biased_loader(
        item["bias"], item["strength"], batch_size=128, verbose=False
    )
    model = train_network(
        train_loader,
        item["bias"],
        item["strength"],
        NAME,
        BATCH_SIZE,
        load=True,
        retrain=False,
        learning_rate=item["learning_rate"],
        epochs=1,
    )

    path_rect, vsr = gm.heatmaps(model, item["bias"], 0)
    path_ell, vse = gm.heatmaps(model, item["bias"], 300000)
    res["images"] = [path_rect, path_ell]
    res["vmax_min"] = [vsr, vse]
    return (name, res)


def compute_all():
    with open("results_cluster.json", "r") as f:
        accuracies = json.load(f)

    gm = GroundTruthMeasures(binary=True)
    for akey in accuracies.keys():
        item = accuracies[akey]
        (name, result) = train_model_evaluate(akey, item, gm)
        accuracies[name] = result

        with open("small_res.json", "w") as f:
            json.dump(accuracies, f, indent=2)


if __name__ == "__main__":
    compute_all()
