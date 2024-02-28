import numpy as np
import os
import json

BIASES = list(np.round(np.linspace(0, 1, 51), 3))
#BIASES = list(np.round(np.linspace(0, 1, 21), 3))
#ITERATIONS = range(10)

def to_name(b, i):
    return "b{}-i{}".format(
        str(round(b, 2)).replace(".", "_"),
        str(i),
    )


def compute_all():
    with open("parallel_accuracies.json", "r") as f:
        accuracies = json.load(f)
    for bias in BIASES:
        makes_sense = any(
            [
                accuracies[to_name(bias, seed)]["train_accuracy"][2] < 90
                for seed in range(0, 5)
            ]
        )
        if makes_sense:
            os.system(
                f"sbatch -J m_{int(bias*100)}_5 ./batch_script_iterations.sh {bias} 5 10"
            )
    for bias in BIASES:
        makes_sense = any(
            [
                accuracies[to_name(bias, seed)]["train_accuracy"][2] < 90
                for seed in range(5, 10)
            ]
        )
        if makes_sense:
            os.system(
                f"sbatch -J m_{int(bias*100)}_10 ./batch_script_iterations.sh {bias} 5 10"
            )
        
    """ for bias in BIASES:
        os.system(
            f"sbatch -J m_{int(bias*100)}_16 ./batch_script_iterations.sh {bias} 10 16"
        ) """


if __name__ == "__main__":
    compute_all()
