import numpy as np
import json
from network import train_network, accuracy_per_class
from biased_noisy_dataset import get_biased_loader

# from crp_attribution import CRPAttribution
import argparse

LEARNING_RATE = 0.001
STRENGTH = 0.5
EPOCHS = 3
BATCH_SIZE = 128
NAME = "models/noisy"
FV_NAME = "fv_model"
IMAGE_PATH = "images/"


def to_name(b, i):
    return "b{}-i{}".format(
        str(round(b, 2)).replace("0.", "0_"),
        str(i),
    )


def train_model_evaluate(*args):
    (bias, strength, allwm, nowm, lr, load_model, num_it) = args
    res = {}
    name = to_name(bias, num_it)
    train_loader = get_biased_loader(
        bias, strength, batch_size=128, verbose=False, img_path=IMAGE_PATH
    )
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
    )

    test_loader = get_biased_loader(
        bias, strength, batch_size=128, verbose=False, split=0.05, img_path=IMAGE_PATH
    )
    res["train_accuracy"] = list(accuracy_per_class(model, test_loader))
    res["all_wm_accuracy"] = list(accuracy_per_class(model, allwm))
    res["no_wm_accuracy"] = list(accuracy_per_class(model, nowm))
    res["bias"] = bias
    res["strength"] = strength
    res["learning_rate"] = lr
    return (name, res)


def compute_with_param(bias):
    with open("parallel_accuracies.json", "r") as f:
        accuracies = json.load(f)
    allwm = get_biased_loader(
        0.0, 0.0, batch_size=128, verbose=False, split=0.05, img_path=IMAGE_PATH
    )
    nowm = get_biased_loader(
        0.0, 1.0, batch_size=128, verbose=False, split=0.05, img_path=IMAGE_PATH
    )
    for num_it in range(4):
        name = to_name(bias, num_it)
        if not (name in accuracies and accuracies[name]["train_accuracy"][2] > 80):
            load_model = False if (name in accuracies) else True
            (name, result) = train_model_evaluate(
                bias, STRENGTH, allwm, nowm, LEARNING_RATE, load_model, num_it
            )
            print(f"(((({name}:{result}))))")
            accuracies[name] = result
        else:
            print(f"(((({name}:{accuracies[name]}))))")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("script_parallel")
    parser.add_argument("bias", help="bias float", type=float)
    args = parser.parse_args()
    compute_with_param(args.bias)
