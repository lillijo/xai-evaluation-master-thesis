import numpy as np
import os

BIASES = list(np.round(np.linspace(0, 1, 21), 3))
ITERATIONS = range(10)


def compute_all():
    for bias in BIASES:
        """ os.system(
            f"sbatch -J m_{int(bias*100)} ./batch_script_iterations.sh {bias} 0 5"
        ) """
        os.system(
            f"sbatch -J m_{int(bias*100)} ./batch_script_iterations.sh {bias} 5 10"
        )


if __name__ == "__main__":
    compute_all()
