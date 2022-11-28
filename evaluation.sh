#!/bin/bash
#SBATCH --partition=gpu-ffa                             # partition you want to run job in
#SBATCH --mem=64G                                # memory resource per cpu
#SBATCH --time=12:00:00					                # time limit
#SBATCH --gpus=1                                        # num of gpus
#SBATCH --cpus-per-task=16                              # cpus per tasks
#SBATCH --job-name="jidenn_eval"                        # change to your job name
#SBATCH --output=./out/%x.%A.%a.log          

ch-run -w -c /home/jankovys/JIDENN /home/jankovys/cuda -- python3 evaluation.py "$@" 


# srun  --partition=gpu-ffa  --mem=64000 --gpus=1  --cpus-per-task=6  ch-run -w -c /home/jankovys/JIDENN /home/jankovys/cuda -- python3 runMe.py

