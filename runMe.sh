#!/bin/bash

# logdir="logs/$(date +"%Y-%m-%d__%H-%M-%S")"

# RES=$(sbatch --parsable train.sh params.logdir=$logdir "$@" ) && sbatch --dependency=afterok:${RES} evaluation.sh base_logdir=$logdir

# trans_logdir="logs/$(date +"%Y-%m-%d__%H-%M-%S")/transformer"
# highway_logdir="logs/$(date +"%Y-%m-%d__%H-%M-%S")/highway"
# part_logdir="logs/$(date +"%Y-%m-%d__%H-%M-%S")/part"
part_logdir="/home/jankovys/JIDENN/logs/2022-11-28__20-37-42/part"
highway_logdir="/home/jankovys/JIDENN/logs/2022-11-28__20-37-42/highway"
trans_logdir="/home/jankovys/JIDENN/logs/2022-11-28__20-37-42/transformer"

# RES=$(sbatch -w volta05 --parsable train.sh params.logdir=$part_logdir params.model=part "$@" ) && sbatch --dependency=afterok:${RES} evaluation.sh base_logdir=$part_logdir model=part
RES=$(sbatch -w volta05 --parsable train.sh params.logdir=$trans_logdir params.model=transformer "$@" ) && sbatch --dependency=afterok:${RES} evaluation.sh base_logdir=$trans_logdir model=transformer
RES=$(sbatch --parsable train.sh params.logdir=$highway_logdir params.model=highway "$@" ) && sbatch --dependency=afterok:${RES} evaluation.sh base_logdir=$highway_logdir model=highway
