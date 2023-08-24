import os
import argparse
import logging
# from functools import partial
# import tensorflow as tf
logging.basicConfig(format='[%(asctime)s][%(levelname)s] - %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
#

from jidenn.data.JIDENNDataset import JIDENNDataset
# from jidenn.preprocess.flatten_dataset import flatten_dataset


parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, help="Path to save the dataset")
parser.add_argument("--load_path", type=str, help="Path to the root file")
parser.add_argument("--backend", type=str, default='pd', help="Backend to use for loading the dataset")
parser.add_argument("--train_frac", type=float, default=0.898, help="Fraction of the dataset to use for training")
parser.add_argument("--dev_frac", type=float, default=0.1, help="Fraction of the dataset to use for development")
parser.add_argument("--test_frac", type=float, default=0.002, help="Fraction of the dataset to use for testing")
parser.add_argument("--reference_variable", type=str, default='jets_PartonTruthLabelID',
                    help="Variable to use as reference for flattening")
parser.add_argument("--wanted_values", type=int, nargs='+',
                    default=[1, 2, 3, 4, 5, 6, 21], help="Values to keep in the reference variable")
parser.add_argument("--shuflle", type=int, default=100_000, required=False, help="Shuffle buffer size")
parser.add_argument("--num_shards", type=int, default=4, required=False,
                    help="Number of shards to save the dataset in")


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.save_path, exist_ok=True)
    dataset = JIDENNDataset.from_root_file(args.load_path, backend=args.backend)
    print(dataset.element_spec)

    dss = dataset.split_train_dev_test(args.train_frac, args.dev_frac, args.test_frac)

    for name, ds in zip(['train', 'dev', 'test'], dss):
        # ds = ds.apply(partial(flatten_dataset, reference_variable=args.reference_variable,
        #                       wanted_values=args.wanted_values))
        # ds = ds.apply(lambda x: x.shuffle(args.shuflle).prefetch(tf.data.AUTOTUNE))
        ds.save(os.path.join(args.save_path, name), num_shards=args.num_shards)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
