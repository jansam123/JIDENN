#!/bin/bash
#SBATCH --partition=ucjf                             
#SBATCH --mem=120G                   
#SBATCH --time=3-12:00:00	                               
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1       
#SBATCH --array 28,29,64,65,82-85,87,107,109,118-135,137-158%4
#SBATCH --cpus-per-task=1                              
#SBATCH --job-name="convert_root"                    
#SBATCH --output=./out/%x.%j.%a.log                             

IFS=$'\n' read -d '' -r -a files < new_root_files.txt
IFS=$'\n' read -d '' -r -a save_path < new_save_paths.txt

echo "Processing file ${files[$SLURM_ARRAY_TASK_ID]}"
echo "Saving to ${save_path[$SLURM_ARRAY_TASK_ID]}"
venv/bin/python3 convert_root.py --file_path=${files[$SLURM_ARRAY_TASK_ID]} --save_path=${save_path[$SLURM_ARRAY_TASK_ID]} 
echo "Done"



