import numpy as np
import json

from crp_attribution import CRPAttribution
from crp_hierarchies import sample_from_categories
from network import train_network, performance_analysis, accuracy_per_class
from biased_dsprites_dataset import get_test_dataset, get_biased_loader

BIASES = np.linspace(0.4, 1.0, 31)
STRENGTHS = [0.5]  # [0.3, 0.5, 0.7]

BATCH_SIZE = 128
NAME = "models/m"
FV_NAME = "fv_model"
ITEMS_PER_CLASS = 245760

accuracies = {}


def to_name(b, s):
    return "data-{}-{}".format(str(b).replace("0.", "0_"), str(s).replace("0.", "0_"))


def from_name(name: str):
    [_, b, s] = name.split("-")
    b = float(b.replace("0_", "0."))
    s = float(s.replace("0_", "0."))
    return b, s


def main():
    accuracies = {}
    _, unb_long, test_loader = get_test_dataset()
    indices = sample_from_categories(unb_long)
    allwm = get_biased_loader(0.0, 0.0, batch_size=128, verbose=False)
    nowm = get_biased_loader(0.0, 1.0, batch_size=128, verbose=False)
    for bias in BIASES:
        for strength in STRENGTHS:
            name = "data-{}-{}".format(
                str(bias).replace("0.", "0_"), str(strength).replace("0.", "0_")
            )
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