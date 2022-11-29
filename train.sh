#!/bin/bash
#SBATCH --partition=gpu-ffa                             # partition you want to run job in
#SBATCH --mem=64G                                # memory resource per cpu
#SBATCH --time=12:00:00					                # time limit
#SBATCH --gpus=2                                        # num of gpus
#SBATCH --nodes=1                                       # num of nodes
#SBATCH --cpus-per-task=16                              # cpus per tasks
#SBATCH --job-name="jidenn_train"                        # change to your job name
#SBATCH --output=./out/%x.%A.%a.log          

# -w volta05                             # partition you want to run job in

# export TF_CPP_MIN_LOG_LEVEL=2

ch-run -w -c /home/jankovys/JIDENN /home/jankovys/cuda -- python3 train.py "$@"




