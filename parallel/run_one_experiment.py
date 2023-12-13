import numpy as np
import json
from network import train_network, accuracy_per_class
from biased_noisy_dataset import get_biased_loader, BiasedNoisyDataset
from ground_truth_measures import GroundTruthMeasures
from torch.utils.data import DataLoader, random_split

# from crp_attribution import CRPAttribution
import argparse

LEARNING_RATE = 0.001
STRENGTH = 0.5
EPOCHS = 3
BATCH_SIZE = 128
NAME = "models/model"
FV_NAME = "fv_model"
IMAGE_PATH = "images/"
LAYER_NAME = "linear_layers.0"

def to_name(b, i):
    return "b{}-i{}".format(
        str(round(b, 2)).replace(".", "_"),
        str(i),
    )


def train_model_evaluate(bias, strength, allwm, nowm, lr, load_model, num_it):
    res = {}
    name = to_name(bias, num_it)
    ds = BiasedNoisyDataset(bias, strength, img_path=IMAGE_PATH)
    trainds, testds, _ = random_split(ds, [0.3, 0.03, 0.67])
    train_loader = DataLoader(trainds, batch_size=128, shuffle=True)
    model = train_network(
        train_loader,
        bias,
        strength,
        NAME,
        BATCH_SIZE,
        load=load_model,
        retrain=False,
        learning_rate=lr,
        epochs=EPOCHS,
        num_it=num_it,
    )

    test_loader = DataLoader(testds, batch_size=128, shuffle=True)
    res["train_accuracy"] = list(accuracy_per_class(model, test_loader))
    res["all_wm_accuracy"] = list(accuracy_per_class(model, allwm))
    res["no_wm_accuracy"] = list(accuracy_per_class(model, nowm))

    res["bias"] = bias
    res["strength"] = strength
    res["learning_rate"] = lr
    
    gm = GroundTruthMeasures(ds)
    flipvalues = gm.intervened_attributions(model, LAYER_NAME)
    ols_vals = gm.ordinary_least_squares(flipvalues)
    mean_logit = gm.mean_logit_change(flipvalues)
    flip_pred = gm.intervened_predictions(model)
    ols_pred = gm.ordinary_least_squares_prediction(flip_pred)
    mean_logit_pred = gm.mean_logit_change_prediction(flip_pred)
    prediction_flip = gm.prediction_flip(flip_pred).tolist()

    res["crp_ols"] = ols_vals
    res["crp_mlc"] = mean_logit.tolist()
    res["pred_ols"] = [a[0] for a in ols_pred]
    res["pred_mlc"] = mean_logit_pred.tolist()
    res["pred_flip"] = prediction_flip

    return (name, res)


def compute_with_param(bias, num_it):
    with open("parallel_accuracies.json", "r") as f:
        accuracies = json.load(f)
    allwm = get_biased_loader(
        0.0, 0.0, batch_size=128, verbose=False, split=0.03, img_path=IMAGE_PATH
    )
    nowm = get_biased_loader(
        0.0, 1.0, batch_size=128, verbose=False, split=0.03, img_path=IMAGE_PATH
    )
    
    name = to_name(bias, num_it)
    if not (name in accuracies and accuracies[name]["train_accuracy"][2] > 80):
        load_model = False if (name in accuracies) else True
        (name, result) = train_model_evaluate(
            bias, STRENGTH, allwm, nowm, LEARNING_RATE, load_model, num_it
        )
        print(f"(((({name}:{result}))))")
        accuracies[name] = result
    else:
        print(f"(((({name}:{accuracies[name]}))))")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("script_parallel")
    parser.add_argument("bias", help="bias float", type=float)
    parser.add_argument("num_it", help="num_it integer", type=int)
    args = parser.parse_args()
    compute_with_param(args.bias, args.num_it)
