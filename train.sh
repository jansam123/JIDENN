#!/bin/bash
#SBATCH --partition=gpu-ffa                             # partition you want to run job in
#SBATCH --mem-per-cpu=4G                                # memory resource per cpu
#SBATCH --time=12:00:00					                # time limit
#SBATCH --gpus=1                                        # num of gpus
#SBATCH --cpus-per-task=16                              # cpus per tasks
#SBATCH --job-name="jidenn_train"                        # change to your job name
#SBATCH --output=./outputs/%x.%j.log               
#SBATCH --mail-user=samueljankovych@gmail.com           # send email when job changes state to email address user@example.com
#SBATCH --mail-type=END,FAIL

# export TF_CPP_MIN_LOG_LEVEL=2

ch-run -w -c /home/jankovys/JIDENN /home/jankovys/cuda -- python3 train.py "$@"


#d

