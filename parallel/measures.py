import numpy as np
import torch
import json
from network import load_model
from biased_noisy_dataset import get_biased_loader, BiasedNoisyDataset
from ground_truth_measures import GroundTruthMeasures
from test_dataset import TestDataset
from torch.utils.data import DataLoader, random_split

# from crp_attribution import CRPAttribution
import argparse

LEARNING_RATE = 0.001
STRENGTH = 0.5
EPOCHS = 3
BATCH_SIZE = 128
NAME = "models/model_seeded"
FV_NAME = "fv_model"
IMAGE_PATH = "images/"
LAYER_NAME = "convolutional_layers.6"#"linear_layers.0"
SEED = 431
ITERATIONS = range(10)


def to_name(b, i):
    return "b{}-i{}".format(
        str(round(b, 2)).replace(".", "_"),
        str(i),
    )


def logit_change_evaluate(item, ds):
    res = item
    ln = "linear" if LAYER_NAME == "linear_layers.0" else "conv"
    if not "pred_ols" in item:
        bias = item["bias"]
        num_it = item["num_it"]
        model = load_model(NAME, bias, num_it)

        gm = GroundTruthMeasures(ds)
        flip_pred = gm.intervened_predictions(model)
        ols_pred = gm.ordinary_least_squares_prediction(flip_pred)
        mean_logit_pred = gm.mean_logit_change_prediction(flip_pred)
        prediction_flip = gm.prediction_flip(flip_pred).tolist()
        res["pred_ols"] = [a[0] for a in ols_pred]
        res["pred_mlc"] = mean_logit_pred.tolist()
        res["pred_flip"] = prediction_flip

    if not f"crp_ols_{ln}" in item:
        bias = item["bias"]
        num_it = item["num_it"]
        model = load_model(NAME, bias, num_it)
        gm = GroundTruthMeasures(ds)
        flipvalues = gm.intervened_attributions(model, LAYER_NAME, disable=True)
        ols_vals = gm.ordinary_least_squares(flipvalues)
        mean_logit = gm.mean_logit_change(flipvalues)
        res[f"crp_ols_{ln}"] = ols_vals
        res[f"crp_mlc_{ln}"] = mean_logit.tolist()

    if not f"rma_mlc_{ln}" in item:
        bias = item["bias"]
        num_it = item["num_it"]
        model = load_model(NAME, bias, num_it)

        gm = GroundTruthMeasures(ds)
        vals_rma, vals_rra = gm.bounding_box_collection(model, LAYER_NAME, disable=True)
        mrc_rma = gm.mean_logit_change(vals_rma, n_latents=1)
        mrc_rra = gm.mean_logit_change(vals_rra, n_latents=1)
        res[f"rma_mlc_{ln}"] = mrc_rma.tolist()
        res[f"rra_mlc_{ln}"] = mrc_rra.tolist()

    return res


def compute_with_param(bias, start_it, end_it):
    with open("parallel_accuracies.json", "r") as f:
        accuracies = json.load(f)
    ds = BiasedNoisyDataset(bias, STRENGTH, img_path=IMAGE_PATH)
    for num_it in range(start_it, end_it):
        name = to_name(bias, num_it)
        if not name in accuracies:
            print(name, "not found")
        else:
            result = logit_change_evaluate(accuracies[name], ds)
            print(f"(((({name}:{result}))))")
            accuracies[name] = result
    """ with open(f"accuracies_{bias}_{end_it}.json", "w") as f:
        json.dump(accuracies, f, indent=2) """


if __name__ == "__main__":
    parser = argparse.ArgumentParser("script_parallel")
    parser.add_argument("bias", help="bias float", type=float)
    parser.add_argument("start_it", help=" float", type=int)
    parser.add_argument("end_it", help=" float", type=int)
    args = parser.parse_args()
    compute_with_param(args.bias, args.start_it, args.end_it)
