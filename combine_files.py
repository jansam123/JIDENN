import tensorflow as tf
import os
import argparse
import pickle
import logging
import shutil
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, List, Tuple, Union
from functools import partial
logging.basicConfig(level=logging.INFO)

from jidenn.data.JIDENNDataset import JIDENNDataset
from jidenn.preprocess.resampling import write_new_variable
from jidenn.preprocess.flatten_dataset import flatten_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, default='data/pythia', help="Path to save the dataset")
parser.add_argument("--load_path", type=str, default='data/pythia/JZ1', help="Path to the root file")
parser.add_argument("--dataset_type", type=str, default='test', help="Dataset type to load")
parser.add_argument("--num_shards", type=int, default=256, help="Number of shards to use when saving the dataset")
parser.add_argument("--jz", type=int, default=2, help="JZ slice to use.")
parser.add_argument("--jz_description_file", type=str,
                    default='data/JZ_description.csv', help="JZ description csv file.")
parser.add_argument("--reference_variable", type=str, default='jets_PartonTruthLabelID',
                    help="Variable to use as reference for flattening")
parser.add_argument("--wanted_values", type=int, nargs='+',
                    default=[1, 2, 3, 4, 5, 6, 21], help="Values to keep in the reference variable")


@tf.function
def get_labeled_pt(data):
    parton = data['jets_PartonTruthLabelID']
    if tf.equal(parton, tf.constant(21)):
        label = 0
    elif tf.reduce_any(tf.equal(parton, tf.constant([1, 2, 3, 4, 5, 6], dtype=tf.int32))):
        label = 1
    else:
        label = -1
    return {**data, 'label': label, 'all_label': parton}


def write_weights(cross_section: float = 1., filt_eff: float = 1., lumi: float = 1., norm: float = 1.) -> Callable:
    @tf.function
    def _calculate_weights(data):
        w = tf.cast(data['weight_mc'], tf.float64)
        cast_lumi = tf.cast(lumi, tf.float64)
        cast_cross_section = tf.cast(cross_section, tf.float64)
        cast_filt_eff = tf.cast(filt_eff, tf.float64)
        cast_norm = tf.cast(norm, tf.float64)
        weight = w[0] * cast_lumi * cast_cross_section * cast_filt_eff / cast_norm
        return {**data, 'weight': weight}
    return _calculate_weights


def main(args: argparse.Namespace) -> None:
    logging.info(f'Running with args: {{{", ".join([f"{k}: {v}" for k, v in vars(args).items()])}}}')
    os.makedirs(args.save_path, exist_ok=True)
    # jz_description = pd.read_csv(args.jz_description_file)

    # filt_eff = jz_description.iloc[args.jz]['filtEff']
    # cross_section = jz_description.iloc[args.jz]['crossSection [pb]']
    # lumi = 139_000

    files = [os.path.join(args.load_path, file, args.dataset_type) for file in os.listdir(
        os.path.join(args.load_path)) if file.startswith('_') and len(os.listdir(os.path.join(args.load_path, file))) > 0]
    dataset = JIDENNDataset.load_parallel(files)

    # norm = dataset.metadata['sum_AOD_w']
    # dataset = dataset.remap_data(write_new_variable(args.jz))
    # dataset = dataset.remap_data(write_weights(cross_section, filt_eff, lumi, norm))
    # if args.dataset_type != 'test':
    #     dataset = dataset.apply(partial(flatten_dataset, reference_variable=args.reference_variable,
    #                                     wanted_values=args.wanted_values))
    #     dataset = dataset.apply(lambda x: x.shuffle(100_000))
    dataset = dataset.apply(lambda x: x.prefetch(tf.data.AUTOTUNE))
    dataset.save(os.path.join(args.save_path, args.dataset_type), num_shards=args.num_shards)

    dataset = JIDENNDataset.load(os.path.join(args.save_path, args.dataset_type))
    print(dataset.element_spec)
    ds_size = dataset.length
    print(f'Dataset size: {ds_size:,}')
    print(
        f"Number of jets: {dataset.dataset.reduce(tf.constant(0, dtype=tf.int32), lambda x, y: x + tf.cast(y['jets_n'], tf.int32)).numpy():,}")
    print('DONE')


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
