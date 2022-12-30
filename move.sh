#!/bin/bash
#SBATCH --partition=ffa                             # partition you want to run job in
#SBATCH --time=12:00:00					                # time limit
#SBATCH --job-name="copy_data"                        # change to your job name
#SBATCH --output=./out/%x.%j.%a.log               

cp -R /home/jankovys/JIDENN/data/dataset2_3  /troja/home/jankovys/JIDENN/data/