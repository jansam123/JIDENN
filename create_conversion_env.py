from typing import Tuple
import os
import argparse
import logging
import uproot
import pandas as pd
logging.basicConfig(format='[%(asctime)s][%(levelname)s] - %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
#

# from jidenn.preprocess.flatten_dataset import flatten_dataset


parser = argparse.ArgumentParser()
parser.add_argument("--load_path", type=str,
                    help="Path to search the root files.")
parser.add_argument("--save_path", type=str,
                    help="Path to save the new datasets.")
parser.add_argument("--events_per_job", type=int,
                    help="Number of jobs to run.")
parser.add_argument("--env_file", type=str,
                    help="Env file to store the variables.")
parser.add_argument("--sample_desc_csv", type=str, default='/home/jankovys/JIDENN/data/sample_description.csv',
                    help="Path to the sample description csv file.")


def include_file_from_new_filename(new_filename: str) -> bool:
    return 'H7EG_jetjet_Cluster_dipole' in new_filename


def include_file_from_old_filename(old_filename: str) -> bool:
    return 'r10724' in old_filename


def create_new_filename(old_filepath: str) -> Tuple[str, str]:
    fi_id = old_filepath.split('/')[-1].split('.')[-2]
    run_number = old_filepath.split('/')[-2].split('.')[3]
    return run_number, fi_id


def main(args: argparse.Namespace) -> None:
    run_desc = pd.read_csv(args.sample_desc_csv, index_col=0)
    run_desc.index = run_desc.index.astype(str)
    run_desc['Description'] = run_desc['Description'].apply(
        lambda x: x[:x.rfind('_')])
    run_desc['JZ'] = 'JZ' + run_desc['JZ'].astype(str)
    

    num_subjobs_per_file = []
    num_entries_per_file = []
    new_file_names = []
    old_file_names = []
    job_names = []
    sample_names = []
    jz_numbers = [] 
    fi_ids = []
    unique_sample_name_jz = []
    for root, _, files in os.walk(args.load_path):
        for file in files:
            if file.endswith('.root'):
                old_file = os.path.join(root, file)
                if not include_file_from_old_filename(old_file):
                    continue
                
                run_number, fi_id = create_new_filename(old_file)
                if run_number in run_desc.index:
                    sample_name = run_desc.loc[run_number, 'Description']
                    jz_number = run_desc.loc[run_number, 'JZ']
                    
                else:
                    logging.warning(
                        f'Run number {run_number} not found in the csv file, using the original run number {run_number}.')
                    sample_name = None
                    jz_number = None
                        
                job_name = f'{sample_name}_{jz_number}{fi_id}'
                if sample_name is not None and jz_number is not None:
                    new_file = os.path.join(args.save_path, sample_name, jz_number, fi_id)
                else:
                    new_file = os.path.join(args.save_path, run_number, fi_id)
                
                
                if not include_file_from_new_filename(new_file):
                    continue

                with uproot.open(old_file) as f:
                    num_entries = f['NOMINAL'].num_entries

                num_jobs = num_entries // args.events_per_job
                if num_entries % args.events_per_job != 0:
                    num_jobs += 1

                num_subjobs_per_file.append(num_jobs - 1)
                num_entries_per_file.append(num_entries)
                new_file_names.append(new_file)
                job_names.append(job_name)
                sample_names.append(sample_name) if sample_name is not None else sample_names.append(run_number)
                jz_numbers.append(jz_number) if jz_number is not None else jz_numbers.append('None')
                fi_ids.append(fi_id)
                old_file_names.append(old_file)
                unique_sample_name_jz.append((sample_name, jz_number))
                
    unique_sample_name_jz = list(set(unique_sample_name_jz))
    unique_sample_names = [x[0] for x in unique_sample_name_jz]
    unique_jz_numbers = [x[1] for x in unique_sample_name_jz]
    num_unique_samples = len(unique_sample_name_jz)

    # create a .env file
    if os.path.exists(args.env_file):
        os.system(f'rm {args.env_file}')
    os.makedirs(os.path.dirname(args.env_file), exist_ok=True)
    with open(args.env_file, 'w') as f:
        f.write(f'NUM_FILES={len(new_file_names)}\n')
        f.write(f'NUM_ENTRIES={sum(num_entries_per_file)}\n')
        f.write(f'NUM_SUBJOBS={sum(num_subjobs_per_file)}\n')
        f.write(
            f'NUM_SUBJOBS_PER_FILE=({" ".join([str(x) for x in num_subjobs_per_file])})\n')
        f.write(
            f'NUM_ENTRIES_PER_FILE=({" ".join([str(x) for x in num_entries_per_file])})\n')
        f.write(f'JOB_NAMES=({" ".join(job_names)})\n')
        f.write(f'OUTPUT_FILES=({" ".join(new_file_names)})\n')
        f.write(f'INPUT_FILES=({" ".join(old_file_names)})\n')
        f.write(f'SAMPLE_NAMES=({" ".join(sample_names)})\n')
        f.write(f'JZ_NUMBERS=({" ".join(jz_numbers)})\n')
        f.write(f'ROOT_FILES_IDS=({" ".join(fi_ids)})\n')
        f.write(f'UNIQUE_SAMPLE_NAMES=({" ".join(unique_sample_names)})\n')
        f.write(f'UNIQUE_JZ_NUMBERS=({" ".join(unique_jz_numbers)})\n')
        f.write(f'NUM_UNIQUE_SAMPLES={num_unique_samples}\n')

    logging.info(f'Created .env file at {args.env_file}')
    logging.info(f'Number of files: {len(old_file_names)}')
    logging.info(f'Number of entries: {sum(num_entries_per_file)}')
    logging.info(f'Number of subjobs: {sum(num_subjobs_per_file)}')
    logging.info(f'DONE')


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
