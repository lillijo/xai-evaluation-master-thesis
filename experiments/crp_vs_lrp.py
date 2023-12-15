import numpy as np
import json
from tqdm import tqdm

from expbasics.ground_truth_measures import GroundTruthMeasures
from expbasics.network import load_model
from expbasics.helper import get_model_etc

IMAGE_PATH = "../dsprites-dataset/images/"  # "images/"
LAYER_NAME = "convolutional_layers.6"  # "linear_layers.0"  #


def logit_change_evaluate(item):
    res = {"bias": item["bias"], "num_it": item["num_it"]}
    model, gm, crp_attribution, test_ds, test_loader = get_model_etc(
        res["bias"], res["num_it"]
    )
    # add change / new measure to compute here
    everything = gm.bounding_box_collection(model, LAYER_NAME, "bbox_all", disable=True)

    everything_rma = [[a[0], a[1], a[2]["rma"], a[3], a[4]] for a in everything]
    everything_rra = [[a[0], a[1], a[2]["rra"], a[3], a[4]] for a in everything]

    mrc_rra = gm.mean_logit_change(everything_rra, n_latents=1)
    mrc_rma = gm.mean_logit_change(everything_rma, n_latents=1)
    res["mrc_rra"] = mrc_rra.tolist()
    res["mrc_rma"] = mrc_rma.tolist()
    return res


def compute_with_param():
    with open("outputs/model_accuracies.json", "r") as f:
        accuracies = json.load(f)
    importances = {}
    for name, item in (pbar := tqdm(accuracies.items())):
        pbar.set_postfix(name=name)
        result = logit_change_evaluate(item)
        importances[name] = result

        with open("outputs/crp_vs_lrp_importances.json", "w") as f:
            json.dump(importances, f, indent=2)


if __name__ == "__main__":
    compute_with_param()
