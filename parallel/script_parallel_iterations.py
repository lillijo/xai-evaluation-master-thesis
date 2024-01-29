import numpy as np
import os
import time

BIASES = list(np.round(np.linspace(0, 1, 51), 3))
#BIASES = list(np.round(np.linspace(0, 1, 21), 3))
#ITERATIONS = range(10)


def compute_all():
    for bias in BIASES:
        os.system(
            f"sbatch -J m_{int(bias*100)}_5 ./batch_script_iterations.sh {bias} 0 5"
        )
    time.sleep(10)
    for bias in BIASES:
        os.system(
            f"sbatch -J m_{int(bias*100)}_10 ./batch_script_iterations.sh {bias} 5 10"
        )
    time.sleep(10)
    for bias in BIASES:
        os.system(
            f"sbatch -J m_{int(bias*100)}_16 ./batch_script_iterations.sh {bias} 10 16"
        )


if __name__ == "__main__":
    compute_all()
