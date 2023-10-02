import numpy as np
from crp_attribution import CRPAttribution
from crp_hierarchies import sample_from_categories
from network import train_network, performance_analysis, accuracy_per_class
from biased_dsprites_dataset import get_test_dataset, get_biased_loader
import json

BIASES = np.round(np.linspace(0.8, 1, 41), 3) # np.round(np.linspace(0, 0.4, 21), 2)
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


def main():
    with open("accuracies.json", "r") as f:
        accuracies = json.load(f)
    _, unb_long, test_loader = get_test_dataset()
    indices = sample_from_categories(unb_long)
    allwm = get_biased_loader(0.0, 0.0, batch_size=128, verbose=False)
    nowm = get_biased_loader(0.0, 1.0, batch_size=128, verbose=False)
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
                load=False,
                retrain=False,
            )
            crp_attribution = CRPAttribution(model, unb_long, FV_NAME, strength, bias)
            accuracies[name] = {}
            accuracies[name]["train_accuracy"] = list(
                accuracy_per_class(model, train_loader)
            )
            accuracies[name]["test_accuracy"] = list(
                accuracy_per_class(model, test_loader)
            )
            accuracies[name]["all_wm_accuracy"] = list(accuracy_per_class(model, allwm))
            accuracies[name]["no_wm_accuracy"] = list(accuracy_per_class(model, nowm))
            print(accuracies[name])
            accuracies[name][
                "watermark_concepts"
            ] = crp_attribution.watermark_concept_importances(indices, unb_long)
            accuracies[name][
                "watermark_mask_concepts"
            ] = crp_attribution.watermark_mask_importance(indices, unb_long)
            accuracies[name]["bias"] = bias
            accuracies[name]["strength"] = strength

            with open("accuracies.json", "+w") as f:
                json.dump(accuracies, f, indent=2)


if __name__ == "__main__":
    main()
