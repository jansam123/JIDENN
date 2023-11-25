import os
import argparse
import logging
from functools import partial
import tensorflow as tf
import pandas as pd
import numpy as np
logging.basicConfig(format='[%(asctime)s][%(levelname)s] - %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
#

from jidenn.data.JIDENNDataset import JIDENNDataset


parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, help="Path to save the dataset")
parser.add_argument("--load_path", type=str, help="Path to the root file")
parser.add_argument("-w", "--weight_var", type=str, default='weight_spectrum', help="Name of the weight variable")
parser.add_argument("--norm", type=int, default=0,
                    help="Normalization factor. Use 0 to normalize to the number of jets.")
parser.add_argument("--num_shards", type=int, default=256, required=False,
                    help="Number of shards to save the dataset in")
parser.add_argument("--weight_sum", type=float, default=None, required=False,
                    help="Number of shards to save the dataset in")


def renorm_weight(weight_var, norm):
    @tf.function
    def _renorm_weights(data):
        w = tf.cast(data[weight_var], tf.float32)
        cast_norm = tf.cast(norm, w.dtype)
        weight = w / cast_norm
        return {**data, weight_var: weight}
    return _renorm_weights


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.save_path, exist_ok=True)
    dataset = JIDENNDataset.load(args.load_path)
    size = dataset.length
    old_norm = args.weight_sum if args.weight_sum is not None else dataset.dataset.reduce(
        np.float32(0), lambda x, y: x + tf.cast(y[args.weight_var], tf.float32))
    # print(f'Old norm: {old_norm.numpy():,}')
    # new_norm = old_norm / tf.cast(args.norm, tf.float32) if args.norm != 0 else old_norm / tf.cast(size, tf.float32)
    new_norm = old_norm / tf.cast(size, tf.float32)
    dataset = dataset.remap_data(renorm_weight(args.weight_var, new_norm))
    dataset.save(args.save_path, num_shards=args.num_shards)

    dataset = JIDENNDataset.load(args.save_path)
    new_norm = dataset.dataset.reduce(np.float32(0), lambda x, y: x + tf.cast(y[args.weight_var], tf.float32))
    print(f'New norm: {new_norm.numpy():,}')
    print(f"Number of jets: {size}")
    dataset.plot_single_variable('jets_pt',
                                 weight_variable='weight_spectrum',
                                 save_path=os.path.join(args.save_path, 'pt_new_W.png'),
                                 badge_text='$N_{\mathrm{jets}}$ = ' +
                                 f'{size:,} \n' + f'$\sum w$ = {new_norm.numpy():,}',
                                 bins=100,
                                 multiple='layer',
                                 hue_variable='jets_PartonTruthLabelID')


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
