#!/bin/bash
#SBATCH --job-name=tester               # Specify job name
#SBATCH --output=tester.o%j             # File name for standard output
#SBATCH --partition=gpu                 # Specify partition name
#SBATCH --gpus=1                        # Specify number of GPUs needed for the job
#SBATCH --exclusive                     # https://slurm.schedmd.com/sbatch.html#OPT_exclusive
#SBATCH --mem=0                         # Request all memory available on all nodes
#SBATCH --account=bd1083
#SBATCH --time=01:00:00                 # Set a limit on the total run time
#SBATCH --mail-type=FAIL                # Notify user by email in case of job failure

set -e
ulimit -s 204800

#module load python3
#source ~/.bashrc
#conda activate mt-lilli

# Execute serial programs, e.g.
# run_one_experiment
python3 -u run_one_experiment.py $1 $2 $3 