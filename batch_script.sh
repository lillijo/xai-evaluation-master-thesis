#!/bin/bash
#SBATCH --job-name=fine_grained         # Specify job name
#SBATCH --output=fine_grained.o%j       # File name for standard output
#SBATCH --partition=gpu                 # Specify partition name
#SBATCH --gpus=4                        # Specify number of GPUs needed for the job
#SBATCH --exclusive                     # https://slurm.schedmd.com/sbatch.html#OPT_exclusive
#SBATCH --mem=0                         # Request all memory available on all nodes
#SBATCH --account=bd1083
#SBATCH --time=08:00:00                 # Set a limit on the total run time
#SBATCH --mail-type=FAIL                # Notify user by email in case of job failure

set -e
ulimit -s 204800

module load python3
source ~/.bashrc
conda activate mt-lilli

# Execute serial programs, e.g.
python -u script_rerun.py