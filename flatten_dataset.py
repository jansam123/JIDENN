import tensorflow as tf
import os
import argparse

from jidenn.preprocess.flatten_dataset import flatten_dataset
from jidenn.preprocess.dataset_ops import save_dataset, load_dataset, split_train_dev_test

parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, help="Path to save the dataset")
parser.add_argument("--load_path", type=str, help="Path to the saved tf.data.Dataset file")
parser.add_argument("--mode", type=str, default='dir_containing_files',
                    help="Mode to load the dataset, either 'single_file' or 'dir_containing_files'")
parser.add_argument("--reference_variable", type=str, default='jets_PartonTruthLabelID',
                    help="Variable to use as reference for flattening")
parser.add_argument("--wanted_values", type=int, nargs='+',
                    default=[1, 2, 3, 4, 5, 6, 21], help="Values to keep in the reference variable")
parser.add_argument("--train_frac", type=float, default=0.8, help="Fraction of the dataset to use for training")
parser.add_argument("--dev_frac", type=float, default=0.1, help="Fraction of the dataset to use for development")
parser.add_argument("--test_frac", type=float, default=0.1, help="Fraction of the dataset to use for testing")
parser.add_argument("--shuflle", type=int, default=100_000, required=False, help="Shuffle buffer size")


def main(args: argparse.Namespace) -> None:

    if args.mode == 'single_file':
        dataset = load_dataset(args.load_path)

    elif args.mode == 'dir_containing_files':
        datasets = []
        for file in os.listdir(args.load_path):
            try:
                datasets.append(load_dataset(os.path.join(args.load_path, file)))
            except Exception as e:
                print(f"Failed to load {file} with error {e}")
                continue
        dataset = tf.data.Dataset.sample_from_datasets(datasets)
    else:
        raise ValueError(f"Mode {args.mode} not supported, use 'single_file' or 'dir_containing_files'")

    dataset = flatten_dataset(dataset, args.reference_variable, args.wanted_values)
    dataset = dataset.shuffle(args.shuflle, seed=42) if args.shuflle is not None else dataset

    if args.train_frac is not None and args.dev_frac is not None and args.test_frac is not None:
        train, dev, test = split_train_dev_test(dataset, args.train_frac, args.dev_frac, args.test_frac)
        save_dataset(train, os.path.join(args.save_path, 'train'))
        save_dataset(dev, os.path.join(args.save_path, 'dev'))
        save_dataset(test, os.path.join(args.save_path, 'test'))
    else:
        save_dataset(dataset, args.save_path)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
