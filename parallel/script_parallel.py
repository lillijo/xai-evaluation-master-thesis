import numpy as np
import os
from tqdm import tqdm
from functools import partialmethod

BIASES = np.round(np.linspace(0, 1, 4), 3) #101
STRENGTHS = [0.5] 

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


def compute_all():
    for bias in BIASES:
        for strength in STRENGTHS:
            os.system(f"sbatch -J models_{bias} ./batch_script_parallel.sh {bias} {strength}")


if __name__ == "__main__":
    compute_all()
