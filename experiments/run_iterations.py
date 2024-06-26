import numpy as np
import torch
import json
import argparse
from torch.utils.data import DataLoader, random_split

from network import train_network, accuracy_per_class
from wdsprites_dataset import BackgroundDataset, BiasedNoisyDataset
from test_dataset import TestDataset
from helper import to_name

LEARNING_RATE = 0.001
STRENGTH = 0.5
EPOCHS = 3
BATCH_SIZE = 128
NAME = "models/background"
FV_NAME = "fv_model"
IMAGE_PATH = "images/"
SEED = 431
EXPERIMENT = "pattern" # "watermark"



def train_model_evaluate(
    bias,
    allwm,
    nowm,
    load_model,
    num_it,
    test_loader,
    train_loader,
):
    res = {}
    name = to_name(bias, num_it)
    model = train_network(
        train_loader,
        bias,
        STRENGTH,
        NAME,
        BATCH_SIZE,
        load=load_model,
        retrain=True,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        num_it=num_it,
        seeded=True,
    )

    res["train_accuracy"] = list(accuracy_per_class(model, test_loader))
    res["all_wm_accuracy"] = list(accuracy_per_class(model, allwm))
    res["no_wm_accuracy"] = list(accuracy_per_class(model, nowm))

    res["bias"] = bias
    res["strength"] = STRENGTH
    res["learning_rate"] = LEARNING_RATE
    res["num_it"] = num_it

    return (name, res)


def compute_with_param(bias, start_it, end_it):
    with open("parallel_accuracies.json", "r") as f:
        accuracies = json.load(f)
    rand_gen = torch.Generator().manual_seed(SEED)

    allwm_dataset = TestDataset(
        2000,
        bias=0.0,
        strength=0.0,
        img_path=IMAGE_PATH,
        experiment=EXPERIMENT
    )
    allwm = DataLoader(allwm_dataset, batch_size=128, shuffle=True, generator=rand_gen)

    nowm_dataset = TestDataset(
        2000,
        bias=0.0,
        strength=1.0,
        img_path=IMAGE_PATH,
        experiment=EXPERIMENT
    )
    nowm = DataLoader(nowm_dataset, batch_size=128, shuffle=True, generator=rand_gen)
    if EXPERIMENT == "pattern":
        ds = BackgroundDataset(bias, STRENGTH, img_path=IMAGE_PATH)
    else:
        ds = BiasedNoisyDataset(bias, STRENGTH, img_path=IMAGE_PATH)
    trainds, testds, _ = random_split(ds, [0.1, 0.01, 0.89], generator=rand_gen)
    train_loader = DataLoader(trainds, batch_size=128, shuffle=True, generator=rand_gen)
    test_loader = DataLoader(testds, batch_size=128, shuffle=True, generator=rand_gen)
    for num_it in range(start_it, end_it):
        name = to_name(bias, num_it)
        if not (name in accuracies and accuracies[name]["train_accuracy"][2] > 90):
            load_model = True if (name in accuracies) else False
            (name, result) = train_model_evaluate(
                bias,
                allwm,
                nowm,
                load_model,
                num_it,
                test_loader,
                train_loader,
            )
            print(f"(((({name}:{result}))))")
            accuracies[name] = result
        else:
            print("model exists:")
            print(f"(((({name}:{accuracies[name]}))))")



def compute_retrain_seeds(bias):
    with open("parallel_accuracies.json", "r") as f:
        accuracies = json.load(f)
    rand_gen = torch.Generator().manual_seed(SEED)

    allwm_dataset = TestDataset(
        2000,
        bias=0.0,
        strength=0.0,
        img_path=IMAGE_PATH,
        experiment=EXPERIMENT
    )
    allwm = DataLoader(allwm_dataset, batch_size=128, shuffle=True, generator=rand_gen)

    nowm_dataset = TestDataset(
        2000,
        bias=0.0,
        strength=1.0,
        img_path=IMAGE_PATH,
        experiment=EXPERIMENT
    )
    nowm = DataLoader(nowm_dataset, batch_size=128, shuffle=True, generator=rand_gen)
    if EXPERIMENT == "pattern":
        ds = BackgroundDataset(bias, STRENGTH, img_path=IMAGE_PATH)
    else:
        ds = BiasedNoisyDataset(bias, STRENGTH, img_path=IMAGE_PATH)
    trainds, testds, _ = random_split(ds, [0.1, 0.01, 0.89], generator=rand_gen)
    train_loader = DataLoader(trainds, batch_size=128, shuffle=True, generator=rand_gen)
    test_loader = DataLoader(testds, batch_size=128, shuffle=True, generator=rand_gen)
    for num_it in [5, 9, 14, 15]:
        name = to_name(bias, num_it)
        if not (name in accuracies and accuracies[name]["train_accuracy"][2] > 80):
            load_model = False
            (name, result) = train_model_evaluate(
                bias,
                allwm,
                nowm,
                load_model,
                num_it,
                test_loader,
                train_loader,
            )
            print(f"(((({name}:{result}))))")
            accuracies[name] = result
        else:
            print("model exists:")
            print(f"(((({name}:{accuracies[name]}))))")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("script_parallel")
    parser.add_argument("bias", help="bias float", type=float)
    parser.add_argument("start_it", help=" float", type=int)
    parser.add_argument("end_it", help=" float", type=int)
    args = parser.parse_args()

    compute_with_param(args.bias, args.start_it, args.end_it)
    # compute_retrain_seeds(args.bias)
