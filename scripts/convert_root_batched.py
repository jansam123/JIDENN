import os
import sys
sys.path.append(os.getcwd())
import argparse
import uproot
import logging
logging.basicConfig(format='[%(asctime)s][%(levelname)s] - %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
#
from jidenn.data.JIDENNDataset import JIDENNDataset

# from jidenn.preprocess.flatten_dataset import flatten_dataset


parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, help="Path to save the dataset.")
parser.add_argument("--load_path", type=str, help="Path to the root file.")
parser.add_argument("--step_size", type=int,
                    help="Number of events to process at once.")
parser.add_argument("--n_parallel", type=int,
                    help="Number of parallel processes to use.")
parser.add_argument("--num_shards", type=int, default=10,
                    help="Number of parallel processes to use.")


def main(args: argparse.Namespace) -> None:

    if os.path.exists(args.save_path):
        # rm dir and its content
        os.system(f'rm -rf {args.save_path}')
        
    tmp_folder = os.path.join(args.save_path, 'tmp')

    with uproot.open(args.load_path) as f:
        num_entries = f['NOMINAL'].num_entries

    logging.info(f'Processing file {args.load_path}')
    logging.info(f'Saving to {args.save_path}')
    logging.info(
        f'Processing entries {num_entries} with step size {args.step_size}')

    dataset = JIDENNDataset.from_root_file_batched(filename=args.load_path,
                                                   tmp_folder=tmp_folder,
                                                   step_size=args.step_size,
                                                   n_parallel=args.n_parallel,
                                                   manual_cast_int=['photons', 'muons'])

    os.makedirs(args.save_path, exist_ok=True)
    dataset.save(args.save_path, num_shards=args.num_shards)
    os.system(f'rm -rf {tmp_folder}')
    logging.info(f'Done')


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
