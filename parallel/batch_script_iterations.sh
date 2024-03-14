#!/bin/bash
#SBATCH --output=more_seeds.o%j         # File name for standard output
#SBATCH --partition=gpu                 # Specify partition name
#SBATCH --gpus=1                        # Specify number of GPUs needed for the job
#SBATCH --exclusive                     # https://slurm.schedmd.com/sbatch.html#OPT_exclusive
#SBATCH --mem=0                         # Request all memory available on all nodes
#SBATCH --account=bd1083
#SBATCH --time=02:00:00                 # Set a limit on the total run time
#SBATCH --mail-type=FAIL                # Notify user by email in case of job failure

set -e
ulimit -s 204800

#module load python3
#source ~/.bashrc
#conda activate mt-lilli

# Execute serial programs, e.g.
# run_one_experiment run_test
python3 -u run_iterations.py $1 $2 $3
#measures.py $1 $2 $3