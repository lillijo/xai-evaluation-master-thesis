import numpy as np
import json
from tqdm import tqdm

from crp_attribution_binary import CRPAttribution
from crp_hierarchies import sample_from_categories
from network_binary import train_network, accuracy_per_class
from biased_dsprites_dataset import get_test_dataset, get_biased_loader
from ground_truth_measures import GroundTruthMeasures

BIASES = [0.04, 0.06, 0.22]
STRENGTHS = [0.5]  # [0.3, 0.5, 0.7]
LEARNING_RATE = 0.0008
EPOCHS = 3
LEARNING_RATES = [
    0.001,
    0.008,
    0.1,
]  # np.round(np.linspace(0.008, 0.017, 10), 3)
BATCH_SIZE = 128
NAME = "models/fine"
FV_NAME = "fv_model"
ITEMS_PER_CLASS = 245760
OLD_LR = 0.016

def to_name(b, s, l):
    return "b{}-s{}-l{}".format(
        str(round(b, 2)).replace("0.", "0_"),
        str(round(s, 2)).replace("0.", "0_"),
        str(round(l, 2)).replace("0.", "0_"),
    )


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


def train_model_evaluate(*args):
    (bias, strength, unb_long, indices, allwm, nowm, gm, lr) = args
    res = {}
    name = to_name(bias, strength, lr)
    print(name)
    train_loader = get_biased_loader(bias, strength, batch_size=128, verbose=False)
    model = train_network(
        train_loader,
        bias,
        strength,
        NAME,
        BATCH_SIZE,
        load=True,
        retrain=False,
        learning_rate=lr,
        epochs=EPOCHS,
    )

    crp_attribution = CRPAttribution(model, unb_long, FV_NAME, strength, bias)

    test_loader = get_biased_loader(
        bias, strength, batch_size=128, verbose=False, split=0.1
    )
    res["train_accuracy"] = list(accuracy_per_class(model, test_loader))
    res["all_wm_accuracy"] = list(accuracy_per_class(model, allwm))
    res["no_wm_accuracy"] = list(accuracy_per_class(model, nowm))
    res["bias"] = bias
    res["strength"] = strength
    res["learning_rate"] = lr
    pred_flip_all = compute_multiple_flips(model, gm)
    res["prediction_flip"] = pred_flip_all
    res["watermark_mask_concepts"] = crp_attribution.watermark_mask_importance(
        indices, unb_long
    )
    return (name, res)


def compute_all():
    with open("accuracies_fine.json", "r") as f:
        accuracies = json.load(f)
    _, unb_long, test_loader = get_test_dataset()
    indices = sample_from_categories(unb_long)
    allwm = get_biased_loader(0.0, 0.0, batch_size=128, verbose=False, split=0.1)
    nowm = get_biased_loader(0.0, 1.0, batch_size=128, verbose=False, split=0.1)
    gm = GroundTruthMeasures(binary=True)

    for bias in BIASES:
        for strength in STRENGTHS:
            for lr in LEARNING_RATES:
                name = to_name(bias, strength, OLD_LR)
                if not (
                    name in accuracies
                    and accuracies[name]["train_accuracy"][0] > 80
                    and accuracies[name]["train_accuracy"][1] > 80
                ):
                    (name, result) = train_model_evaluate(
                        bias, strength, unb_long, indices, allwm, nowm, gm, lr
                    )
                    accuracies[name] = result

        with open("accuracies_fine.json", "w") as f:
            json.dump(accuracies, f, indent=2)


if __name__ == "__main__":
    compute_all()
