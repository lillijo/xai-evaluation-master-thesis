import numpy as np
import os

BIASES = np.round(np.linspace(0, 1, 11), 3) # 101


def compute_all():
    for bias in BIASES:
        os.system(f"sbatch -J models_{bias} ./batch_script_parallel.sh {bias}")


if __name__ == "__main__":
    compute_all()
