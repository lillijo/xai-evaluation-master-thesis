import numpy as np
import os

BIASES = list(np.round(np.linspace(0.1, 0.4, 4), 3)) + list(
    np.round(np.linspace(0.5, 1, 51), 3)
)
# np.round(np.linspace(0.8, 1, 21), 3) # 101


def compute_all():
    for bias in BIASES:
        os.system(f"sbatch -J ratio_{int(bias*100)} ./batch_script_parallel.sh {bias}")


if __name__ == "__main__":
    compute_all()
