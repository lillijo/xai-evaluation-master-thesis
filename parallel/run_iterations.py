import numpy as np
import torch
import json
from network import train_network, accuracy_per_class
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
LAYER_NAME = "linear_layers.0"
SEED = 431
ITERATIONS = range(10)


def to_name(b, i):
    return "b{}-i{}".format(
        str(round(b, 2)).replace(".", "_"),
        str(i),
    )


def train_model_evaluate(
    bias,
    strength,
    allwm,
    nowm,
    lr,
    load_model,
    num_it,
    rand_gen,
    testds,
    train_loader,
):
    res = {}
    name = to_name(bias, num_it)
    model = train_network(
        train_loader,
        bias,
        strength,
        NAME,
        BATCH_SIZE,
        load=load_model,
        retrain=False,
        learning_rate=lr,
        epochs=EPOCHS,
        num_it=num_it,
        seeded=True,
    )

    test_loader = DataLoader(testds, batch_size=128, shuffle=True, generator=rand_gen)
    res["train_accuracy"] = list(accuracy_per_class(model, test_loader))
    res["all_wm_accuracy"] = list(accuracy_per_class(model, allwm))
    res["no_wm_accuracy"] = list(accuracy_per_class(model, nowm))

    res["bias"] = bias
    res["strength"] = strength
    res["learning_rate"] = lr

    """ gm = GroundTruthMeasures(ds)
    flipvalues = gm.intervened_attributions(model, LAYER_NAME, disable=True)
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
    res["pred_flip"] = prediction_flip """

    return (name, res)


def compute_with_param(bias, start_it, end_it):
    accuracies = {}
    """ with open("parallel_accuracies.json", "r") as f:
        accuracies = json.load(f) """
    rand_gen = torch.Generator().manual_seed(SEED)

    allwm_dataset = TestDataset(
        10000,
        bias=0.0,
        strength=0.0,
        im_dir="allwm",
        img_path=IMAGE_PATH,
    )
    allwm = DataLoader(allwm_dataset, batch_size=128, shuffle=True, generator=rand_gen)

    nowm_dataset = TestDataset(
        10000,
        bias=0.0,
        strength=1.0,
        im_dir="nowm",
        img_path=IMAGE_PATH,
    )
    nowm = DataLoader(nowm_dataset, batch_size=128, shuffle=True, generator=rand_gen)
    ds = BiasedNoisyDataset(bias, STRENGTH, img_path=IMAGE_PATH)
    trainds, testds, _ = random_split(ds, [0.2, 0.03, 0.77], generator=rand_gen)
    train_loader = DataLoader(trainds, batch_size=128, shuffle=True, generator=rand_gen)
    for num_it in range(start_it, end_it):
        name = to_name(bias, num_it)
        if not (name in accuracies and accuracies[name]["train_accuracy"][2] > 80):
            load_model = False if (name in accuracies) else True
            (name, result) = train_model_evaluate(
                bias,
                STRENGTH,
                allwm,
                nowm,
                LEARNING_RATE,
                load_model,
                num_it,
                rand_gen,
                testds,
                train_loader,
            )
            print(f"(((({name}:{result}))))")
            accuracies[name] = result
        else:
            print("model exists:")
            print(f"(((({name}:{accuracies[name]}))))")
    """ with open(f"accuracies_{bias}_{end_it}.json", "w") as f:
        json.dump(accuracies, f, indent=2) """
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser("script_parallel")
    parser.add_argument("bias", help="bias float", type=float)
    parser.add_argument("start_it", help=" float", type=int)
    parser.add_argument("end_it", help=" float", type=int)
    args = parser.parse_args()
    compute_with_param(args.bias, args.start_it, args.end_it)
