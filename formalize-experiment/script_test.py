import numpy as np
import json
from tqdm import tqdm

from crp_attribution_binary import CRPAttribution
from crp_hierarchies import sample_from_categories
from network_binary import train_network, accuracy_per_class
from biased_dsprites_dataset import get_test_dataset, get_biased_loader
from ground_truth_measures import GroundTruthMeasures

BIASES = [0.5]#np.round(np.linspace(0.8, 1, 41), 3)
STRENGTHS = [0.5]  # [0.3, 0.5, 0.7]

BATCH_SIZE = 128
NAME = "models/fine"
FV_NAME = "fv_model"
ITEMS_PER_CLASS = 245760

accuracies = {}


def to_name(b, s):
    return "data-{}-{}".format(
        str(round(b, 2)).replace("0.", "0_"), str(round(s, 2)).replace("0.", "0_")
    )


def from_name(name):
    [_, b, s] = name.split("-")
    b = float(b.replace("0_", "0."))
    s = float(s.replace("0_", "0."))
    return b, s

def compute_multiple_flips(model, gm):
    count = 0
    pred_flip_all = {
        "watermark": 0.0,
        "shape": 0.0,
        "scale": 0.0,
        "orientation": 0.0,
        "posX": 0.0,
        "posY": 0.0,
    }

    for i in tqdm(range(0, 737280, 3943)):
        pred_flip = gm.prediction_flip(i, model)
        count += 1
        for k in pred_flip_all.keys():
            pred_flip_all[k] += pred_flip[k]
    for k in pred_flip_all.keys():
        pred_flip_all[k] = pred_flip_all[k] / count
    return pred_flip_all

def main():
    with open("accuracies_fine.json", "r") as f:
        accuracies = json.load(f)
    _, unb_long, test_loader = get_test_dataset()
    indices = sample_from_categories(unb_long)
    allwm = get_biased_loader(0.0, 0.0, batch_size=128, verbose=False)
    nowm = get_biased_loader(0.0, 1.0, batch_size=128, verbose=False)
    gm = GroundTruthMeasures(binary=True)
    for bias in BIASES:
        for strength in STRENGTHS:
            name = to_name(bias, strength)
            # if name not in accuracies:
            print(name)
            train_loader = get_biased_loader(bias, strength, batch_size=128)
            model = train_network(
                train_loader,
                bias,
                strength,
                NAME,
                BATCH_SIZE,
                load=True,
                retrain=False,
            )
            crp_attribution = CRPAttribution(model, unb_long, FV_NAME, strength, bias)
            accuracies[name] = {}
            """ accuracies[name]["train_accuracy"] = list(
                accuracy_per_class(model, train_loader)
            )
            accuracies[name]["all_wm_accuracy"] = list(accuracy_per_class(model, allwm))
            accuracies[name]["no_wm_accuracy"] = list(accuracy_per_class(model, nowm))

            accuracies[name]["bias"] = bias
            accuracies[name]["strength"] = strength """
            pred_flip_all = compute_multiple_flips(model, gm)
            accuracies[name]["prediction_flip"] = pred_flip_all
            print(accuracies[name])
            accuracies[name][
                "watermark_mask_concepts"
            ] = crp_attribution.watermark_mask_importance(indices, unb_long)

            with open("accuracies_fine.json", "+w") as f:
                json.dump(accuracies, f, indent=2)


if __name__ == "__main__":
    main()
