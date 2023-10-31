import numpy as np
import json
from network import train_network, accuracy_per_class
from biased_dsprites_dataset import get_biased_loader, BiasedDSpritesDataset
from crp_attribution import CRPAttribution

BIASES = np.round(np.linspace(0, 1, 101), 3)
STRENGTHS = [0.5]  # [0.3, 0.5, 0.7]  #
LEARNING_RATE = [0.009, 0.007, 0.005, 0.001]
EPOCHS = 3
BATCH_SIZE = 128
NAME = "models/nmf"
FV_NAME = "fv_model"


def to_name(b, s, l):
    return "b{}-s{}-l{}".format(
        str(round(b, 2)).replace("0.", "0_"),
        str(round(s, 2)).replace("0.", "0_"),
        str(round(l, 4)).replace("0.", "0_"),
    )


def train_model_evaluate(*args):
    (bias, strength, allwm, nowm, unbiased_ds, lr) = args
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
        load=False,
        retrain=False,
        learning_rate=lr,
        epochs=EPOCHS,
    )

    test_loader = get_biased_loader(
        bias, strength, batch_size=128, verbose=False, split=0.05
    )
    res["train_accuracy"] = list(accuracy_per_class(model, test_loader))
    res["all_wm_accuracy"] = list(accuracy_per_class(model, allwm))
    res["no_wm_accuracy"] = list(accuracy_per_class(model, nowm))
    # crp_attribution = CRPAttribution(model, unbiased_ds, "nmf", strength, bias)
    # crp_attribution.compute_feature_vis()
    res["bias"] = bias
    res["strength"] = strength
    res["learning_rate"] = lr
    return (name, res)


def compute_all():
    with open("model_accuracies.json", "r") as f:
        accuracies = json.load(f)
    allwm = get_biased_loader(0.0, 0.0, batch_size=128, verbose=False, split=0.05)
    nowm = get_biased_loader(0.0, 1.0, batch_size=128, verbose=False, split=0.05)
    unbiased_ds = BiasedDSpritesDataset(
        verbose=False,
        bias=0.0,
        strength=0.5,
    )

    for bias in BIASES:
        for strength in STRENGTHS:
            for learnr in LEARNING_RATE:
                name = to_name(bias, strength, learnr)
                if not (name in accuracies and accuracies[name]["train_accuracy"][2] > 80):
                    (name, result) = train_model_evaluate(
                        bias, strength, allwm, nowm, unbiased_ds, learnr
                    )
                    accuracies[name] = result

        with open("model_accuracies.json", "w") as f:
            json.dump(accuracies, f, indent=2)


if __name__ == "__main__":
    compute_all()
