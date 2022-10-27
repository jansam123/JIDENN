#!/bin/bash

for model in basic_fc highway
do
    for activation in relu gelu elu silu
    do
        echo  "runMe.py hydra.run.dir='logs/grid_search/{SLURM_ARRAY_TASK_ID}' params.activation=${activation} params.model=${model}" >> jobs.txt
    done
done

for batch_size in 512 1024
do
    echo  "runMe.py hydra.run.dir='logs/grid_search/{SLURM_ARRAY_TASK_ID}' dataset.batch_size=${batch_size}" >> jobs.txt
done

for shuffle in 100 10000
do
    echo  "runMe.py hydra.run.dir='logs/grid_search/{SLURM_ARRAY_TASK_ID}' dataset.shuffle_buffer=${shuffle}" >> jobs.txt
done

for norm in 1000 100000
do
    echo  "runMe.py hydra.run.dir='logs/grid_search/{SLURM_ARRAY_TASK_ID}' preprocess.normalization_size=${norm}" >> jobs.txt
done

for hidden_layers in '"[512, 512]"' '"[512, 512, 512, 512, 512]"' '"[512, 512, 512, 512, 512, 512, 512, 512, 512, 512]"' '"[1024, 1024, 1024]"'
do
    echo  "runMe.py hydra.run.dir='logs/grid_search/{SLURM_ARRAY_TASK_ID}' basic_fc.hidden_layers=${hidden_layers}" >> jobs.txt
done

for num_layers in 2 5 10
do
    echo  "runMe.py hydra.run.dir='logs/grid_search/{SLURM_ARRAY_TASK_ID}' highway.num_layers=${num_layers} params.model=highway" >> jobs.txt
done

echo  "runMe.py hydra.run.dir='logs/grid_search/{SLURM_ARRAY_TASK_ID}' highway.num_layers=3 highway.layer_size=1024 params.model=highway" >> jobs.txt



# IFS=$'\n' read -d '' -r -a lines < jobs.txt