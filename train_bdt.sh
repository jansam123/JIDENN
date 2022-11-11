#!/bin/bash
#SBATCH --partition=ffa                             # partition you want to run job in
#SBATCH --nodelist=dw05                                # memory resource per cpu
#SBATCH --mem=128G                                # memory resource per cpu
#SBATCH --time=12:00:00					                # time limit
#SBATCH --cpus-per-task=40                              # cpus per tasks
#SBATCH --job-name="jidenn_train"                        # change to your job name
#SBATCH --output=./output/train.bdt.%j.log               
#SBATCH --mail-user=samueljankovych@gmail.com           # send email when job changes state to email address user@example.com
#SBATCH --mail-type=END,FAIL  

# export TF_CPP_MIN_LOG_LEVEL=2
logdir="logs/bdt/$(date +"%Y-%m-%d__%H-%M-%S")"


ch-run -w -c /home/jankovys/JIDENN /home/jankovys/cuda -- python3 train.py params.model=BDT preprocess.normalize=False dataset.take=1_000_000 dataset.batch_size=128 params.epochs=1 params.logdir=$logdir preprocess.draw_distribution=0 dataset.test_size=0.01


# srun  --partition=gpu-ffa  --mem=64000 --gpus=1  --cpus-per-task=6  ch-run -w -c /home/jankovys/JIDENN /home/jankovys/cuda -- python3 runMe.py
