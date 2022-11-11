#!/bin/bash
#SBATCH --partition=ffa                             # partition you want to run job in
#SBATCH --mem-per-cpu=4G                                # memory resource per cpu
#SBATCH --time=12:00:00					                # time limit
#SBATCH --cpus-per-task=32                              # cpus per tasks
#SBATCH --job-name="jidenn_eval"                        # change to your job name
#SBATCH --output=./output/%x.%j.log 
#SBATCH --mail-user=samueljankovych@gmail.com           # send email when job changes state to email address user@example.com
#SBATCH --mail-type=END,FAIL

# export TF_CPP_MIN_LOG_LEVEL=2

ch-run -w -c /home/jankovys/JIDENN /home/jankovys/cuda -- python3 evaluation.py "$@" 


# srun  --partition=gpu-ffa  --mem=64000 --gpus=1  --cpus-per-task=6  ch-run -w -c /home/jankovys/JIDENN /home/jankovys/cuda -- python3 runMe.py

