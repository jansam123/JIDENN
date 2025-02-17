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
parser.add_argument("--step_size", type=int, default=100_000,
                    help="Number of events to process at once.")
parser.add_argument("--step_size_str", type=str, default='',
                    help="Number of events to process at once.")
parser.add_argument("--n_parallel", type=int,
                    help="Number of parallel processes to use.")
parser.add_argument("--num_shards", type=int, default=64,
                    help="Number of shards to save the dataset in.")
parser.add_argument("--ttree", type=str, default='NOMINAL', help="TTree to load from the root file")
parser.add_argument("--metadata", type=str, default='h_metadata', help="Metadata histogram to load from the root file")
parser.add_argument("--not_use_ray", action='store_true', help="Disable ray")


def main(args: argparse.Namespace) -> None:

    # if os.path.exists(args.save_path):
    #     # rm dir and its content
    #     os.system(f'rm -rf {args.save_path}')
        
    tmp_folder = os.path.join(args.save_path, 'tmp')
    os.makedirs(tmp_folder, exist_ok=True)
    files_to_skip = [int(f.split('_')[-1]) for f in os.listdir(tmp_folder) if f.startswith('batch')]

    logging.info(f'Processing file {args.load_path}')
    logging.info(f'Saving to {args.save_path}')
    
    step_size = args.step_size_str if args.step_size_str != '' else args.step_size
    
    if '*' not in args.load_path:
        with uproot.open(args.load_path) as f:
            num_entries = f[args.ttree].num_entries
        logging.info(
            f'Processing entries {num_entries} with step size {step_size}')
    use_ray = not args.not_use_ray
    dataset = JIDENNDataset.from_root_file_batched(filename=args.load_path,
                                                   tmp_folder=tmp_folder,
                                                   step_size=step_size,
                                                   n_parallel=args.n_parallel,
                                                   tree_name=args.ttree,
                                                   use_ray=use_ray,
                                                   to_skip=files_to_skip,
                                                   metadata_hist=args.metadata if args.metadata != '' else None,
                                                   manual_cast_int=['photons', 'muons'])

    os.makedirs(args.save_path, exist_ok=True)
    dataset.save(args.save_path, num_shards=args.num_shards)
    os.system(f'rm -rf {tmp_folder}')
    logging.info(f'Done')


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
