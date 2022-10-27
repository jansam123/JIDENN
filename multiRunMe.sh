#!/bin/bash
#SBATCH --partition=gpu-ffa                             # partition you want to run job in
#SBATCH --mem-per-cpu=3G                               # memory resource per cpu
#SBATCH --time=12:00:00					                # time limit
#SBATCH --gpus=1                                        # num of gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1       
#SBATCH --array 23-24
#SBATCH --cpus-per-task=10                              # cpus per tasks
#SBATCH --job-name="jidenn_training"                    # change to your job name
#SBATCH --output=./outputs/output\_%a.log                             # stdout and stderr output file
#SBATCH --mail-user=samueljankovych@gmail.com           # send email when job changes state to email address user@example.com

IFS=$'\n' read -d '' -r -a lines < jobs.txt
export TZ="Europe/Prague"

ch-run -w -c /home/jankovys/JIDENN /home/jankovys/cuda -- python3 ${lines[$SLURM_ARRAY_TASK_ID]} hydra.run.dir="logs/grid_search/${SLURM_ARRAY_TASK_ID}" dataset.take=4000000 

# srun  --partition=gpu-ffa  --mem=64000 --gpus=1  --cpus-per-task=6  ch-run -w -c /home/jankovys/JIDENN ./../cuda -- python3 runMe.py


#highway relu
#basic_fc gelu
#batch_size 512 or 1024
# shuffle 10_000
# norm 100_000
# more layers on basic_fc do not help
# more layers on highway do help