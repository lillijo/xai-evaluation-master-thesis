import numpy as np
import os

BIASES = np.round(np.linspace(0, 1, 101), 3)
STRENGTHS = [0.5]  # [0.3, 0.5, 0.7]  #
LEARNING_RATE = [0.009, 0.007, 0.005, 0.001]

def compute_all():
    for bias in BIASES:
        for strength in STRENGTHS:
            for learnr in LEARNING_RATE:
                os.system(f"sbatch ./batch_script_parallel.sh {bias} {strength} {learnr}")                

if __name__ == "__main__":
    compute_all()
