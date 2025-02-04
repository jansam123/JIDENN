import pandas as pd
import uproot
import logging
import argparse
from typing import Tuple
import os
import sys
sys.path.append(os.getcwd())
logging.basicConfig(format='[%(asctime)s][%(levelname)s] - %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
#

# from jidenn.preprocess.flatten_dataset import flatten_dataset


parser = argparse.ArgumentParser()
parser.add_argument("--load_path", type=str,
                    help="Path to search the root files.")
parser.add_argument("--save_path", type=str,
                    help="Path to save the new datasets.")
parser.add_argument("--env_file", type=str,
                    help="Env file to store the variables.")
parser.add_argument("--sample_desc_csv", type=str, default='/home/jankovys/JIDENN/data/sample_description.csv',
                    help="Path to the sample description csv file.")

# * EDIT THIS FUNCTION TO SELECT THE FILES YOU WANT TO CONVERT


def include_file_from_new_filename(new_filename: str) -> bool:
    return True  # 'H7EG_jetjet_Cluster_dipole' in new_filename

# * EDIT THIS FUNCTION TO SELECT THE FILES YOU WANT TO CONVERT


def include_file_from_old_filename(old_filename: str) -> bool:
    return True  # 'r10724' in old_filename

# * EDIT THIS FUNCTION TO CREATE THE NEW FILENAME


def create_new_filename(old_filepath: str) -> Tuple[str, str ,str]:
    fi_id = old_filepath.split('/')[-1].split('.')[-2]
    e_tag=old_filepath.split('/')[-2].split('.')[5].split('_')[0]
    run_number = old_filepath.split('/')[-2].split('.')[3]
    return run_number, fi_id, e_tag


def main(args: argparse.Namespace) -> None:
    run_desc = pd.read_csv(args.sample_desc_csv, index_col=0)
    run_desc.index = run_desc.index.astype(str)
    # run_desc['JZ'] = 'JZ' + run_desc['JZ'].astype(str)

    num_subjobs_per_file = []
    num_entries_per_file = []
    new_file_names = []
    old_file_names = []
    job_names = []
    sample_names = []
    # jz_numbers = []
    fi_ids = []
    unique_sample_name_jz = []
    for root, _, files in os.walk(args.load_path):
        for file in files:
            if file.endswith('.root'):
                old_file = os.path.join(root, file)
                # logging.info(f'Processing file {old_file}')
                if not include_file_from_old_filename(old_file):
                    continue

                run_number, fi_id, e_tag = create_new_filename(old_file)
                if run_number in run_desc.index:
                    sample_name = run_desc.query(f'eTag == "{e_tag}"').loc[run_number, 'Description']
                    # logging.info(f'Sample name: {sample_name}')
                    # jz_number = run_desc.loc[run_number, 'JZ']
                else:
                    logging.warning(
                        f'Run number {run_number} not found in the csv file, using the original run number {run_number}.')
                    sample_name = None
                    # jz_number = None

                if sample_name is not None:
                    new_file = os.path.join(
                        args.save_path, sample_name, 'tmp', fi_id)
                else:
                    new_file = os.path.join(
                        args.save_path, run_number, 'tmp', fi_id)

                if not include_file_from_new_filename(new_file):
                    continue
    
                new_file_names.append(new_file)
                sample_names.append(
                    sample_name) if sample_name is not None else sample_names.append(run_number)
                fi_ids.append(fi_id)
                old_file_names.append(old_file)
                unique_sample_name_jz.append(sample_name)

    unique_sample_names = list(set(unique_sample_name_jz))
    num_unique_samples = len(unique_sample_names)

    logging.info(f'Creating .env file at {args.env_file}')
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
        # f.write(f'JZ_NUMBERS=({" ".join(jz_numbers)})\n')
        f.write(f'ROOT_FILES_IDS=({" ".join(fi_ids)})\n')
        f.write(f'UNIQUE_SAMPLE_NAMES=({" ".join(unique_sample_names)})\n')
        # f.write(f'UNIQUE_JZ_NUMBERS=({" ".join(unique_jz_numbers)})\n')
        f.write(f'NUM_UNIQUE_SAMPLES={num_unique_samples}\n')

    logging.info(f'Created .env file at {args.env_file}')
    logging.info(f'Number of files: {len(old_file_names)}')
    logging.info(f'Number of entries: {sum(num_entries_per_file)}')
    logging.info(f'Number of subjobs: {sum(num_subjobs_per_file)}')
    logging.info(f'DONE')


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
