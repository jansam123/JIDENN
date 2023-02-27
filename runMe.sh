#!/bin/bash

base_logdir="logs/$(date +"%Y-%m-%d__%H-%M-%S")"
# base_logdir="logs/2023-02-16__20-10-21"

trans_logdir=$base_logdir/transformer
highway_logdir=$base_logdir/highway
basic_fc_logdir=$base_logdir/basic_fc
part_logdir=$base_logdir/part
depart_logdir=$base_logdir/depart
interacting_part_logdir=$base_logdir/interacting_part
interacting_depart_logdir=$base_logdir/interacting_depart
bdt_logdir=$base_logdir/bdt


# RES=$(sbatch -w volta05  --parsable --job-name="dparti_train" train.sh params.logdir=$interacting_depart_logdir params.model=depart models.depart.interaction=True "$@")  && sbatch --dependency=afterok:${RES} --job-name="dparti_eval" evaluation.sh base_logdir=$interacting_depart_logdir model=depart interaction=True
# RES=$(sbatch -w volta05 --parsable --job-name="parti_train"  train.sh params.logdir=$interacting_part_logdir params.model=part models.part.interaction=True "$@" ) && sbatch --dependency=afterok:${RES} --job-name="parti_eval" evaluation.sh base_logdir=$interacting_part_logdir model=part interaction=True
RES=$(sbatch -w volta05 --gpus=4 --parsable --job-name="dpart_train"  train.sh params.logdir=$depart_logdir params.model=depart "$@" ) && sbatch --dependency=afterok:${RES} --job-name="dpart_eval" evaluation.sh base_logdir=$depart_logdir model=depart interaction=True
# RES=$(sbatch -w volta05 --parsable --job-name="part_train"   train.sh params.logdir=$part_logdir params.model=part "$@" ) && sbatch --dependency=afterok:${RES} --job-name="part_eval" evaluation.sh base_logdir=$part_logdir model=part
# RES=$(sbatch -w volta05 --parsable --job-name="trans_train"  train.sh params.logdir=$trans_logdir params.model=transformer "$@" ) && sbatch --dependency=afterok:${RES} --job-name="trans_eval" evaluation.sh base_logdir=$trans_logdir model=transformer
# RES=$(sbatch --job-name="bdt_train" --gpus=0 --cpus-per-task=32 --parsable train.sh params.logdir=$bdt_logdir params.model=bdt preprocess.normalize=False params.epochs=1 models.bdt.num_threads=32 dataset.shuffle_buffer=null "$@") && sbatch --dependency=afterok:${RES} --job-name="bdt_eval" evaluation.sh base_logdir=$bdt_logdir model=bdt
# RES=$(sbatch --parsable --job-name="highwy_train"  train.sh params.logdir=$highway_logdir params.model=highway "$@" ) && sbatch --dependency=afterok:${RES} --job-name="highwy_eval" evaluation.sh base_logdir=$highway_logdir model=highway
# RES=$(sbatch --parsable --job-name="fc_train"  train.sh params.logdir=$basic_fc_logdir params.model=basic_fc "$@" ) && sbatch --dependency=afterok:${RES} --job-name="fc_eval" evaluation.sh base_logdir=$basic_fc_logdir model=basic_fc



# RES=$(sbatch --gpus=1 --job-name="part_train" --parsable train.sh params.logdir=$part_logdir params.model=part "$@" ) #&& sbatch --dependency=afterok:${RES} --job-name="part_eval" evaluation.sh base_logdir=$part_logdir model=part
# RES=$(sbatch --gpus=1 --job-name="dpart_train" --parsable train.sh params.logdir=$depart_logdir params.model=depart "$@" ) #&& sbatch --dependency=afterok:${RES} --job-name="dpart_eval" evaluation.sh base_logdir=$depart_logdir model=depart
# RES=$(sbatch --gpus=1 --parsable --job-name="parti_train" train.sh params.logdir=$interacting_part_logdir params.model=part models.part.interaction=True "$@" ) #&& sbatch --dependency=afterok:${RES} --job-name="parti_eval" evaluation.sh base_logdir=$interacting_part_logdir model=part interaction=True
# RES=$(sbatch --gpus=1 --parsable --job-name="dparti_train" train.sh params.logdir=$interacting_depart_logdir params.model=depart models.depart.interaction=True "$@" ) #&& sbatch --dependency=afterok:${RES} --job-name="dparti_eval" evaluation.sh base_logdir=$interacting_depart_logdir model=depart interaction=True
# RES=$(sbatch --job-name="bdt_train" --gpus=0 --cpus-per-task=32 --parsable train.sh params.logdir=$bdt_logdir params.model=bdt preprocess.normalize=False params.epochs=1 models.bdt.num_threads=32 dataset.shuffle_buffer=null "$@") #&& sbatch --dependency=afterok:${RES} --job-name="bdt_eval" evaluation.sh base_logdir=$bdt_logdir model=bdt
# RES=$(sbatch --gpus=1 --job-name="trans_train" --parsable train.sh params.logdir=$trans_logdir params.model=transformer "$@" ) #&& sbatch --dependency=afterok:${RES} --job-name="trans_eval" evaluation.sh base_logdir=$trans_logdir model=transformer
# RES=$(sbatch --gpus=1 --parsable --job-name="highwy_train"  train.sh params.logdir=$highway_logdir params.model=highway "$@" ) #&& sbatch --dependency=afterok:${RES} --job-name="highwy_eval" evaluation.sh base_logdir=$highway_logdir model=highway
# RES=$(sbatch --gpus=1 --parsable --job-name="fc_train"  train.sh params.logdir=$basic_fc_logdir params.model=basic_fc "$@" ) #&& sbatch --dependency=afterok:${RES} --job-name="fc_eval" evaluation.sh base_logdir=$basic_fc_logdir model=basic_fc