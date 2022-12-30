#!/bin/bash

# logdir="logs/$(date +"%Y-%m-%d__%H-%M-%S")"

# RES=$(sbatch --parsable train.sh params.logdir=$logdir "$@" ) && sbatch --dependency=afterok:${RES} evaluation.sh base_logdir=$logdir

trans_logdir="logs/$(date +"%Y-%m-%d__%H-%M-%S")/transformer"
highway_logdir="logs/$(date +"%Y-%m-%d__%H-%M-%S")/highway"
part_logdir="logs/$(date +"%Y-%m-%d__%H-%M-%S")/part"
depart_logdir="logs/$(date +"%Y-%m-%d__%H-%M-%S")/depart"
interacting_part_logdir="logs/$(date +"%Y-%m-%d__%H-%M-%S")/interacting_part"
bdt_logdir="logs/$(date +"%Y-%m-%d__%H-%M-%S")/bdt"
#part_logdir="/home/jankovys/JIDENN/logs/2022-11-28__20-37-42/part"
#highway_logdir="/home/jankovys/JIDENN/logs/2022-11-28__20-37-42/highway"
#trans_logdir="/home/jankovys/JIDENN/logs/2022-11-28__20-37-42/transformer"

# RES=$(sbatch  --parsable train.sh params.logdir=$part_logdir params.model=part "$@" ) && sbatch --dependency=afterok:${RES} evaluation.sh base_logdir=$part_logdir model=part
RES=$(sbatch -w volta05 --gpus=2  --parsable train.sh params.logdir=$depart_logdir params.model=depart "$@" ) && sbatch --dependency=afterok:${RES} evaluation.sh base_logdir=$depart_logdir model=depart
# RES=$(sbatch --parsable train.sh params.logdir=$interacting_part_logdir params.model=part part.interaction=False "$@" ) && s                                                                                                                                     batch --dependency=afterok:${RES} evaluation.sh base_logdir=$interacting_part_logdir model=part 
# RES=$(sbatch -w volta05 --gpus=0 --cpus-per-task=64 --parsable train.sh params.logdir=$bdt_logdir params.model=bdt preprocess.normalize=False params.epochs=1 bdt.num_threads=64 dataset.shuffle_buffer=null "$@") && sbatch --dependency=afterok:${RES} evaluation.sh base_logdir=$bdt_logdir model=bdt
# RES=$(sbatch  --parsable train.sh params.logdir=$trans_logdir params.model=transformer "$@" ) && sbatch --dependency=afterok:${RES} evaluation.sh base_logdir=$trans_logdir model=transformer
# RES=$(sbatch --parsable   train.sh params.logdir=$highway_logdir params.model=highway "$@" ) && sbatch --dependency=afterok:${RES} evaluation.sh base_logdir=$highway_logdir model=highway
