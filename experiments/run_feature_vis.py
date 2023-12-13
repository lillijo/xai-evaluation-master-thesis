import numpy as np
import json
import torch
from tqdm import tqdm
from expbasics.network import load_model
from crp.visualization import FeatureVisualization
from zennit.composites import EpsilonPlusFlat
from crp.concepts import ChannelConcept
from crp.attribution import CondAttribution

from expbasics.test_dataset import TestDataset


def to_name(b, i):
    return "b{}-i{}".format(
        str(round(b, 2)).replace(".", "_"),
        str(i),
    )


LEARNING_RATE = 0.001
STRENGTH = 0.5
EPOCHS = 3
BATCH_SIZE = 128
NAME = "../clustermodels/model"  # models/noise_pos
FV_NAME = "model"
IMAGE_PATH = "../dsprites-dataset/images/"  # "images/"
LAYER_NAME = "linear_layers.0"  # "convolutional_layers.6"
BIASES = list(np.round(np.linspace(0.0, 0.4, 5), 3)) + list(
    np.round(np.linspace(0.5, 1, 11), 3)
)
ITERATIONS = range(6)


def make_feature_vis(item, test_ds):
    res = item
    name = to_name(item["bias"], item["num_it"])
    model = load_model(NAME, item["bias"], item["num_it"]) 
    path = f"crp-data/{FV_NAME}_{name}_fv"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tdev = torch.device(device)
    attribution = CondAttribution(model, no_param_grad=True, device=tdev)
    cc = ChannelConcept()
    composite = EpsilonPlusFlat()
    fv = FeatureVisualization(attribution, test_ds, {LAYER_NAME: cc}, path=path)
    fv.run(composite, 0, len(test_ds), BATCH_SIZE, checkpoint=500, on_device=tdev)

    return (name, path)


def compute_with_param(bias, num_it):
    with open("outputs/blub_accuracies.json", "r") as f:
        accuracies = json.load(f)
    ds = TestDataset(length=3000)
    name = to_name(bias, num_it)
    (name, path) = make_feature_vis(accuracies[name], ds)
    print(f"(((({name}:{path}))))")
    return name, path


def compute_all():
    paths = {}
    for bias in tqdm(BIASES):
        for num_it in ITERATIONS:
            name, path = compute_with_param(bias, num_it)
            paths[name] = path
        with open("outputs/fv_paths_model.json", "w") as f:
            json.dump(paths, f, indent=1)


if __name__ == "__main__":
    """parser = argparse.ArgumentParser("script_parallel")
    parser.add_argument("bias", help="bias float", type=float)
    parser.add_argument("num_it", help="num iteration int", type=int)
    args = parser.parse_args()
    compute_with_param(args.bias, args.num_it)"""
    compute_all()
