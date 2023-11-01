import numpy as np
import json
from network import train_network, accuracy_per_class
from biased_dsprites_dataset import get_biased_loader, BiasedDSpritesDataset
from crp_attribution import CRPAttribution
import argparse
from tqdm import tqdm

LEARNING_RATE = [0.009, 0.007, 0.005, 0.001]
EPOCHS = 3
BATCH_SIZE = 128
NAME = "models/nmf"
FV_NAME = "fv_model"
IMAGE_PATH = "images/"


def to_name(b, s, l):
    return "b{}-s{}-l{}".format(
        str(round(b, 2)).replace("0.", "0_"),
        str(round(s, 2)).replace("0.", "0_"),
        str(round(l, 4)).replace("0.", "0_"),
    )


def compute_with_param(bias, strength):
    with open("test.json", "r") as f:
        tests = json.load(f)
    for learnr in tqdm(LEARNING_RATE):        
        name = to_name(bias, strength, learnr)
        if not (name in tests and tests[name]["train_accuracy"][2] > 80):
            tests[name] = [bias, strength, learnr]

            with open("test.json", "w") as f:
                json.dump(tests, f, indent=2)
        else:
            print("already existing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("script_parallel")
    parser.add_argument("bias", help="bias float", type=float)
    parser.add_argument("strength", help="strength float", type=float)
    args = parser.parse_args()
    compute_with_param(args.bias, args.strength)
