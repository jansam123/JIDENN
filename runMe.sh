#!/bin/bash
#SBATCH --partition=gpu-ffa                             # partition you want to run job in
#SBATCH --mem-per-cpu=10G                               # memory resource per cpu
#SBATCH --time=12:00:00					                # time limit
#SBATCH --gpus=2                                        # num of gpus
#SBATCH --cpus-per-task=6                               # cpus per tasks
#SBATCH --job-name="jidenn_training"                    # change to your job name
#SBATCH --output=output.log                             # stdout and stderr output file
#SBATCH --mail-user=samueljankovych@gmail.com           # send email when job changes state to email address user@example.com

# export TF_CPP_MIN_LOG_LEVEL=2
ch-run -w -c /home/jankovys/JIDENN /home/jankovys/cuda -- python3 runMe.py 

# srun  --partition=gpu-ffa  --mem=64000 --gpus=1  --cpus-per-task=6  ch-run -w -c /home/jankovys/JIDENN /home/jankovys/cuda -- python3 runMe.py


