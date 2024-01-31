import numpy as np
import os
import time

BIASES = list(np.round(np.linspace(0, 1, 51), 3))
#BIASES = list(np.round(np.linspace(0, 1, 21), 3))


def compute_all():
    for bias in BIASES:
        os.system(
            f"sbatch -J m_{int(bias*100)} ./batch_script_iterations.sh {bias}"
        )


if __name__ == "__main__":
    compute_all()
