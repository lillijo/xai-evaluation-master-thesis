import numpy as np
import os

BIASES = list(np.round(np.linspace(0, 1, 21), 3))
ITERATIONS = range(10)


def compute_all():
    for num_it in ITERATIONS:
        for bias in BIASES:
            os.system(
                f"sbatch -J m_{int(bias*100)}_{num_it} ./batch_script_parallel.sh {bias} {num_it}"
            )


if __name__ == "__main__":
    compute_all()
