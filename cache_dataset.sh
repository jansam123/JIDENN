#!/bin/bash
#SBATCH --partition=ffa                             # partition you want to run job in
#SBATCH --mem=60G                                # memory resource per cpu
#SBATCH --time=12:00:00					                # time limit
#SBATCH --mincpus=32                              # cpus per tasks
#SBATCH --job-name="cache_dataset"                             # change to your job name
#SBATCH --output=./output/%x.%j.log               

# export TF_CPP_MIN_LOG_LEVEL=2

ch-run -w -c /home/jankovys/JIDENN /home/jankovys/cuda -- python3 cache_dataset.py "$@"