#!/bin/bash
#SBATCH --partition=gpu-ffa                             # partition you want to run job in
#SBATCH --mem-per-cpu=6G                                # memory resource per cpu
#SBATCH --time=12:00:00					                # time limit
#SBATCH --gpus=1                                        # num of gpus
#SBATCH --cpus-per-task=16                              # cpus per tasks
#SBATCH --job-name="jidenn"                             # change to your job name
#SBATCH --output=output_highway_small.log                # stdout and stderr output file
#SBATCH --mail-user=samueljankovych@gmail.com           # send email when job changes state to email address user@example.com

# export TF_CPP_MIN_LOG_LEVEL=2
export TZ="Europe/Prague"
ch-run -w -c /home/jankovys/JIDENN /home/jankovys/cuda -- python3 runMe.py highway.num_layers=10 dataset.take=1000000 params.epochs=3

# srun  --partition=gpu-ffa  --mem=64000 --gpus=1  --cpus-per-task=6  ch-run -w -c /home/jankovys/JIDENN /home/jankovys/cuda -- python3 runMe.py
#16 cores and 6GB of RAM per core => 390_000*256 ~ 100_000_000 data points  

