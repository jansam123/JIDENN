#!/bin/bash
#SBATCH --partition=ffa                             
#SBATCH --mem=32G                   
#SBATCH --time=12:00:00	                               
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1       
#SBATCH --array=0-5
#SBATCH --cpus-per-task=32
#SBATCH --job-name="triple_split"                    
#SBATCH --output=out/convert_%A_%a.log
#SBATCH --constraint=avx

# JZ00
# frac of q/g = 0.04
# 56GB total size
# effeciently 0.25GB per shard
# 0.04 * 56GB / 0.25GB ~ 10 shards

# JZ01
# frac of q/g = 0.43
# 110GB total size
# effeciently 0.25GB per shard
# 0.43 * 110GB / 0.25GB ~ 190 shards

# JZ02
# frac of q/g = 0.79
# 53GB total size
# effeciently 0.25GB per shard
# 0.79 * 53GB / 0.25GB ~ 168 shards

# JZ03
# frac of q/g = 0.83
# 69GB total size
# effeciently 0.25GB per shard
# 0.83 * 69GB / 0.25GB ~ 230 shards

# JZ04
# frac of q/g = 0.86
# 82GB total size
# effeciently 0.25GB per shard
# 0.86 * 82GB / 0.25GB ~ 282 shards

# JZ05
# frac of q/g = 0.86
# 47GB total size
# effeciently 0.25GB per shard
# 0.86 * 47GB / 0.25GB ~ 162 shards


num_shards=(10 190 168 230 282 162)

printf -v pad_id "%02d" $SLURM_ARRAY_TASK_ID

ch-run -w -c /home/jankovys/JIDENN /home/jankovys/cuda -- python3 split_dataset.py --load_path="/home/jankovys/JIDENN/data/dataset2/JZ${pad_id}_r10724" --save_path="/home/jankovys/JIDENN/data/dataset2_3/JZ${pad_id}_r10724" --num_shards=${num_shards[$SLURM_ARRAY_TASK_ID]} 