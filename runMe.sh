#!/bin/bash

# logdir="logs/$(date +"%Y-%m-%d__%H-%M-%S")"

# RES=$(sbatch --parsable train.sh params.logdir=$logdir "$@" ) && sbatch --dependency=afterok:${RES} evaluation.sh base_logdir=$logdir
base_logdir="logs/$(date +"%Y-%m-%d__%H-%M-%S")"

trans_logdir=$base_logdir/transformer
highway_logdir=$base_logdir/highway
part_logdir=$base_logdir/part
depart_logdir=$base_logdir/depart
interacting_part_logdir=$base_logdir/interacting_part
interacting_depart_logdir=$base_logdir/interacting_depart
bdt_logdir=$base_logdir/bdt

# cp -r /home/jankovys/JIDENN/src $base_logdir
#
# depart_logdir="/home/jankovys/JIDENN/logs/2023-01-01__09-45-13/depart"
#highway_logdir="/home/jankovys/JIDENN/logs/2022-11-28__20-37-42/highway"
#trans_logdir="/home/jankovys/JIDENN/logs/2022-11-28__20-37-42/transformer"

# RES=$(sbatch -w volta05 --job-name="part_train" --parsable train.sh params.logdir=$part_logdir params.model=part "$@" ) && sbatch --dependency=afterok:${RES} --job-name="part_eval" evaluation.sh base_logdir=$part_logdir model=part
# RES=$(sbatch -w volta05 --gpus=2 --job-name="depart_train" --parsable train.sh params.logdir=$depart_logdir params.model=depart "$@" ) && sbatch --dependency=afterok:${RES} --job-name="depart_eval" evaluation.sh base_logdir=$depart_logdir model=depart
RES=$(sbatch -w volta05 --parsable --job-name="part_train" train.sh params.logdir=$interacting_part_logdir params.model=part models.part.interaction=True "$@" ) && sbatch --dependency=afterok:${RES} evaluation.sh base_logdir=$interacting_part_logdir model=part interaction=True
# RES=$(sbatch -w volta05 --parsable --job-name="depart_train" train.sh params.logdir=$interacting_depart_logdir params.model=depart models.depart.interaction=True "$@" ) && sbatch --dependency=afterok:${RES} evaluation.sh base_logdir=$interacting_depart_logdir model=depart interaction=True
# RES=$(sbatch --job-name="bdt_train" --gpus=0 --cpus-per-task=32 --parsable train.sh params.logdir=$bdt_logdir params.model=bdt preprocess.normalize=False params.epochs=1 bdt.num_threads=32 dataset.shuffle_buffer=null "$@") && sbatch --dependency=afterok:${RES} --job-name="bdt_eval" evaluation.sh base_logdir=$bdt_logdir model=bdt
# RES=$(sbatch -w volta05 --job-name="trans_train" --parsable train.sh params.logdir=$trans_logdir params.model=transformer "$@" ) && sbatch --dependency=afterok:${RES} --job-name="trans_eval" evaluation.sh base_logdir=$trans_logdir model=transformer
# RES=$(sbatch --parsable --job-name="highwy_train"  train.sh params.logdir=$highway_logdir params.model=highway "$@" ) && sbatch --dependency=afterok:${RES} --job-name="highwy_eval" evaluation.sh base_logdir=$highway_logdir model=highway

rm slurm-*.out