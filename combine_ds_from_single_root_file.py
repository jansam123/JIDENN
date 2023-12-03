from jidenn.data.JIDENNDataset import JIDENNDataset
import tensorflow as tf
import os
import argparse
import pickle
import logging
logging.basicConfig(format='[%(asctime)s][%(levelname)s] - %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, help="Path to save the dataset")
parser.add_argument("--load_path", type=str,
                    help="Path to the folder containing saved datasets from SINGLE file")
parser.add_argument("--num_shards", type=int, default=256,
                    help="Number of shards to use when saving the dataset")
parser.add_argument("--num_datasets", type=int, default=256,
                    help="Expected number of datasets to be combined.")

def main(args: argparse.Namespace) -> None:
    logging.info(
        f'Combining datasets from {args.load_path} into {args.save_path}')
    files = [os.path.join(
        args.load_path, file) for file in os.listdir(
        args.load_path) if file.startswith('batch')]
    if len(files) != args.num_datasets + 1:
        logging.warning(
            f'Expected {args.num_datasets + 1} datasets, but found {len(files)}.')

    dataset = JIDENNDataset.load_multiple(
        files, metadata_combiner=lambda x: x[0], mode='interleave')
    logging.info(f'Loaded dataset with metadata: {dataset.metadata}')
    logging.info(f'Loaded dataset of length {dataset.length}')

    logging.info(f'Saving dataset to {args.save_path}')
    dataset = dataset.apply(lambda x: x.prefetch(tf.data.AUTOTUNE))

    os.system(f'rm -rf {args.save_path}') if os.path.exists(args.save_path) else None
    os.makedirs(args.save_path, exist_ok=True)
    dataset.save(args.save_path, num_shards=args.num_shards)
    logging.info(f'Saved dataset to {args.save_path}')

    logging.info(f'DONE')


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
