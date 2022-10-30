#!/bin/bash
#SBATCH --partition=gpu-ffa                             # partition you want to run job in
#SBATCH --mem-per-cpu=4G                                # memory resource per cpu
#SBATCH --time=12:00:00					                # time limit
#SBATCH --gpus=1                                        # num of gpus
#SBATCH --cpus-per-task=32                              # cpus per tasks
#SBATCH --job-name="jidenn_train"                             # change to your job name
#SBATCH --output=./output/%x.%j.log               
#SBATCH --mail-user=samueljankovych@gmail.com           # send email when job changes state to email address user@example.com
#SBATCH --mail-type=END,FAIL

# export TF_CPP_MIN_LOG_LEVEL=2

ch-run -w -c /home/jankovys/JIDENN /home/jankovys/cuda -- python3 train.py "$@"


# srun  --partition=gpu-ffa  --mem=64000 --gpus=1  --cpus-per-task=6  ch-run -w -c /home/jankovys/JIDENN /home/jankovys/cuda -- python3 runMe.py
#16 cores and 6GB of RAM per core => 390_000*256 ~ 100_000_000 data points  

