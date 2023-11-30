import numpy as np
import json
import torch
from expbasics.network import train_network, accuracy_per_class
from expbasics.biased_noisy_dataset import BiasedNoisyDataset
from expbasics.helper import to_name
from torch.utils.data import DataLoader, random_split
from crp.visualization import FeatureVisualization
from zennit.composites import EpsilonPlusFlat
from crp.concepts import ChannelConcept
from crp.attribution import CondAttribution

# from crp_attribution import CRPAttribution
import argparse

LEARNING_RATE = 0.001
STRENGTH = 0.5
EPOCHS = 3
BATCH_SIZE = 128
NAME = "../clustermodels/noise_pos"  # models/noise_pos
FV_NAME = "noise_pos"
IMAGE_PATH = "../dsprites-dataset/images/"  # "images/"
LAYER_NAME = "convolutional_layers.6"
BIASES = list(np.round(np.linspace(0.0, 0.4, 4), 3)) + list(
    np.round(np.linspace(0.5, 1, 51), 3)
)


def make_feature_vis(item, test_loader, test_ds):
    res = item
    name = to_name(item["bias"], item["num_it"])
    model = train_network(
        test_loader,
        item["bias"],
        item["strength"],
        NAME,
        BATCH_SIZE,
        load=True,
        retrain=False,
        learning_rate=item["learning_rate"],
        epochs=EPOCHS,
        num_it=item["num_it"],
    )
    path = f"crp-data/{FV_NAME}_{name}_fv"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tdev = torch.device(device)
    attribution = CondAttribution(model, no_param_grad=True, device=tdev)
    cc = ChannelConcept()
    composite = EpsilonPlusFlat()
    fv = FeatureVisualization(attribution, test_ds, {LAYER_NAME: cc}, path=path)
    fv.run(composite, 0, len(test_ds), 128, 2048, on_device=tdev)

    return (name, path)


def compute_with_param(bias):
    with open("outputs/noise_pos_accuracies.json", "r") as f:
        accuracies = json.load(f)
    ds = BiasedNoisyDataset(
        verbose=False, strength=STRENGTH, bias=0.0, img_path=IMAGE_PATH
    )
    split = 0.05
    [train_ds, test_ds] = random_split(ds, [1 - split, split])
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)

    for num_it in range(4):
        name = to_name(bias, num_it)
        (name, path) = make_feature_vis(accuracies[name], test_loader, test_ds)
        print(f"(((({name}:{path}))))")

def compute_iteration(num_it):
    with open("outputs/noise_pos_accuracies.json", "r") as f:
        accuracies = json.load(f)
    ds = BiasedNoisyDataset(
        verbose=False, strength=STRENGTH, bias=0.0, img_path=IMAGE_PATH
    )
    split = 0.05
    [train_ds, test_ds] = random_split(ds, [1 - split, split])
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)

    for bias in BIASES:
        name = to_name(bias, num_it)
        (name, path) = make_feature_vis(accuracies[name], test_loader, test_ds)
        print(f"(((({name}:{path}))))")

def compute_all():
    with open("outputs/noise_pos_accuracies.json", "r") as f:
        accuracies = json.load(f)
    ds = BiasedNoisyDataset(
        verbose=False, strength=STRENGTH, bias=0.0, img_path=IMAGE_PATH
    )
    split = 0.03
    [train_ds, test_ds] = random_split(ds, [1 - split, split])
    print(len(test_ds))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)
    for name in accuracies.keys():
        (name, path) = make_feature_vis(accuracies[name], test_loader, test_ds)
        accuracies[name]["fv_conv"] = path
        with open("outputs/noise_pos_accuracies.json", "w") as f:
                json.dump(accuracies, f, indent=2)

if __name__ == "__main__":
    """ parser = argparse.ArgumentParser("script_parallel")
    parser.add_argument("bias", help="bias float", type=float)
    args = parser.parse_args()
    compute_with_param(args.bias) """
    compute_all()
    
