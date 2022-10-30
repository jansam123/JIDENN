#!/bin/bash

logdir="logs/$(date +"%Y-%m-%d__%H-%M-%S")"

RES=$(sbatch --parsable train.sh params.logdir=$logdir "$@" ) && sbatch --dependency=afterok:${RES} evaluation.sh base_logdir=$logdir


