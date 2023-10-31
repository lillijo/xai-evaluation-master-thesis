import numpy as np
import json
from network import train_network, accuracy_per_class
from biased_dsprites_dataset import get_biased_loader, BiasedDSpritesDataset
from crp_attribution import CRPAttribution
import argparse


def tester(bias, strength, learnr):
    with open("test.json", "r") as f:
        accuracies = json.load(f)
    name = to_name(bias, strength, learnr)
    print(f"tester name: {name}")
    accuracies[name] = {"bias": bias, "strength": strength, "learnr": learnr}

    with open("test.json", "w") as f:
        json.dump(accuracies, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("tester")
    parser.add_argument("bias", help="bias float", type=float)
    parser.add_argument("strength", help="strength float", type=float)
    parser.add_argument("learnr", help="learning rate float", type=float)
    args = parser.parse_args()
    print(f"args: {args}")
    tester(args.bias, args.strength, args.learnr)
