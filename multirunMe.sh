#!/bin/bash
#SBATCH --partition=gpu-ffa                             # partition you want to run job in
#SBATCH --mem-per-cpu=10G                               # memory resource per cpu
#SBATCH --time=12:00:00					                # time limit
#SBATCH --gpus=1                                        # num of gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1       
#SBATCH --array 0-7
#SBATCH --cpus-per-task=6                              # cpus per tasks
#SBATCH --job-name="jidenn_training"                    # change to your job name
#SBATCH --output=output\_%a.log                             # stdout and stderr output file
#SBATCH --mail-user=samueljankovych@gmail.com           # send email when job changes state to email address user@example.com

# export TF_CPP_MIN_LOG_LEVEL=2
ch-run -w -c /home/jankovys/JIDENN /home/jankovys/cuda -- python3 runMe.py data.subfolder_id=$SLURM_ARRAY_TASK_ID  hydra.run.dir="logs/multirun_PFO/$SLURM_ARRAY_TASK_ID" data.weight=\'weight_mc\[:,0\]*1e$SLURM_ARRAY_TASK_ID\' 
# srun  --partition=gpu-ffa  --mem=64000 --gpus=1  --cpus-per-task=6  ch-run -w -c /home/jankovys/JIDENN ./../cuda -- python3 runMe.py


