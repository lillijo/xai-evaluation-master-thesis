import numpy as np
import os

BIASES =list(np.round(np.linspace(0.0, 0.4, 5), 3)) + list(
    np.round(np.linspace(0.5, 1, 11), 3)
) 
ITERATIONS = range(6)  # range(4, 8)
# np.round(np.linspace(0.8, 1, 21), 3) # 101


def compute_all():
    for bias in BIASES:
        for num_it in ITERATIONS:
            os.system(
                f"sbatch -J m_{int(bias*100)}_{num_it} ./batch_script_parallel.sh {bias} {num_it}"
            )


if __name__ == "__main__":
    compute_all()
