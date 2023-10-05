import numpy as np
import json
from network_binary import train_network, accuracy_per_class
from biased_dsprites_dataset import get_test_dataset, get_biased_loader

BIASES = np.round(np.linspace(0.8, 1, 41), 3)
STRENGTHS = [0.5]  # [0.3, 0.5, 0.7]

def to_name(b, s):
    return "hyper-{}-{}".format(
        str(round(b, 2)).replace("0.", "0_"), str(round(s, 2)).replace("0.", "0_")
    )

def main():
    hyperparameters = []
    _, unb_long, test_loader = get_test_dataset(split=0.1)
    train_loader = get_biased_loader(0.8, 0.5, batch_size=128, verbose=False)
    train_valid = get_biased_loader(0.8, 0.5, batch_size=128, verbose=False, split=0.1)
    for lr in np.round(np.linspace(0.008, 0.017, 10), 5):
        for b in [0.7,0.8]:
            name = to_name(lr, b)
            model = train_network(
                train_loader,
                b,
                0.5,
                name,
                128,
                load=False,
                retrain=False,
                epochs=1,
                learning_rate=lr,
                optim="Adam",
            )
            res = {
                #"test_acc": list(accuracy_per_class(model, test_loader)),
                "train_acc": list(accuracy_per_class(model, train_valid)),
                "optimizer": "Adam",
                "bias": b,
                "learning_rate": float(lr),
                "name": name,
            }
            hyperparameters.append(res)

    with open("hyperparameters.json", "w+") as f:
        json.dump({"hyperparameters": hyperparameters}, f, indent=2)


if __name__ == "__main__":
    main()
