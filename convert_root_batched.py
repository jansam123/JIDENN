from jidenn.data.JIDENNDataset import JIDENNDataset
import os
import argparse
import uproot
import logging
logging.basicConfig(format='[%(asctime)s][%(levelname)s] - %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
#

# from jidenn.preprocess.flatten_dataset import flatten_dataset


parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, help="Path to save the dataset.")
parser.add_argument("--load_path", type=str, help="Path to the root file.")
parser.add_argument("--job_id", type=int, help="Path to the root file.")
parser.add_argument("--step_size", type=int,
                    help="Number of events to process at once.")
parser.add_argument("--n_parallel", type=int,
                    help="Number of parallel processes to use.")
parser.add_argument("--num_shards", type=int, default=1,
                    help="Number of parallel processes to use.")


def main(args: argparse.Namespace) -> None:

    tmp_folder = os.path.join(args.save_path, 'tmp')

    with uproot.open(args.load_path) as f:
        num_entries = f['NOMINAL'].num_entries

    entry_start = args.job_id * args.step_size
    entry_stop = min((args.job_id + 1) * args.step_size, num_entries)
    logging.info(f'Processing file {args.load_path}')
    logging.info(f'Job id: {args.job_id}, num of cpus: {args.n_parallel}')
    logging.info(
        f'Processing entries {entry_start} to {entry_stop} out of {num_entries} with step size {args.step_size}')

    dataset = JIDENNDataset.from_root_file_batched(filename=args.load_path,
                                                   tmp_folder=tmp_folder,
                                                   step_size=args.step_size,
                                                   entry_start=entry_start,
                                                   entry_stop=entry_stop,
                                                   n_parallel=args.n_parallel)

    path = os.path.join(args.save_path, f'batch_{args.job_id}')
    if os.path.exists(path):
        # rm dir and its content
        os.system(f'rm -rf {path}')
    os.makedirs(path, exist_ok=True)
    dataset.save(path, num_shards=args.num_shards)
    logging.info(f'Done')


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
