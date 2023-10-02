import numpy as np
import json
from tqdm import tqdm

from script import to_name
from network import train_network
from biased_dsprites_dataset import get_biased_loader
from ground_truth_measures import GroundTruthMeasures

BIASES = np.round(np.linspace(0, 1, 51), 2)
STRENGTHS = [0.5]  # [0.3, 0.5, 0.7]

BATCH_SIZE = 128
NAME = "models/m"
FV_NAME = "fv_model"
ITEMS_PER_CLASS = 245760

ground_truths = {}

def compute_multiple_flips(model, gm):
    count = 0
    pred_flip_all = {
        "wm": 0.0,
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
    allwm = get_biased_loader(0.0, 0.0, batch_size=128, verbose=False)
    gm = GroundTruthMeasures()
    for bias in BIASES:
        for strength in STRENGTHS:
            name = to_name(bias, strength)
            #if name not in accuracies:
            print(name)
            model = train_network(
                allwm,
                bias,
                strength,
                NAME,
                BATCH_SIZE,
                load=True,
                retrain=False,
            )
            pred_flip_all = compute_multiple_flips(model, gm)
            ground_truths[name] = pred_flip_all

            with open("ground_truths.json", "+w") as f:
                json.dump(ground_truths, f, indent=2)


if __name__ == "__main__":
    main()
