#!/bin/bash

for batch_size in 512 1024
do
    for epochs in 10 20 30
    do
        for num_layers in 10 20 30 40 50
        do
            for layer_size in 256 512 1024
            do
                for activation in relu gelu 
                do
                    echo  "highway.num_layers=${num_layers} params.model=highway higway.layer_size=${layer_size} params.activation=${activation} params.epochs=${epochs} params.decay_steps=$((146208768*epochs/batch_size))" >> jobs.txt
                done
            done
        done
    done
done


for batch_size in 512 1024
do
    for epochs in 10 20 30
    do
        for num_layers in 5 10 20
        do
            for layer_size in 256 512 1024
            do
                for activation in relu gelu 
                do
                    echo  "highway.num_layers=${num_layers} params.model=basic_fc basic_fc.layer_size=${layer_size} params.activation=${activation} params.epochs=${epochs} params.decay_steps=$((146208768*epochs/batch_size))" >> jobs.txt
                done
            done
        done
    done
done
