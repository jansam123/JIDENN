#!/bin/bash
#SBATCH --partition=gpu-ffa                             # partition you want to run job in
#SBATCH --mem=32G                               # memory resource per cpu
#SBATCH --time=12:00:00	                               # time limit
#SBATCH --gpus=1                                        # num of gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1       
#SBATCH --array 159-159
#SBATCH --cpus-per-task=16                              # cpus per tasks
#SBATCH --job-name="jidenn_train"                    # change to your job name
#SBATCH --output=./outputs/%x.%a.log                             # stdout and stderr output file

IFS=$'\n' read -d '' -r -a lines < jobs.txt
export TZ="Europe/Prague"

ch-run -w -c /home/jankovys/JIDENN /home/jankovys/cuda -- python3 train.py ${lines[$SLURM_ARRAY_TASK_ID]} params.logdir="logs/grid_search/${SLURM_ARRAY_TASK_ID}" 

# srun  --partition=gpu-ffa  --mem=64000 --gpus=1  --cpus-per-task=6  ch-run -w -c /home/jankovys/JIDENN ./../cuda -- python3 runMe.py



