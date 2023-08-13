import tensorflow as tf
import os
import argparse
import pickle
import logging
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, List, Tuple, Union
from functools import partial


from jidenn.preprocess.resampling import resample_var_with_labels, write_new_variable, get_cut_fn, get_filter_fn, resample_labels
# from jidenn.evaluation.plotter import plot_single_dist
from jidenn.data.JIDENNDataset import JIDENNDataset


parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, default='data/pythia_flat/train', help="Path to save the dataset")
parser.add_argument("--file_path", type=str, default='data/pythia', help="Path to the root file")
parser.add_argument("--num_shards", type=int, default=256, help="Number of shards to use when saving the dataset")
parser.add_argument("--bins", type=int, default=70, help="Number of bins to use for the binning")
parser.add_argument("--take", type=int, default=None, help="Number of samples to take")
parser.add_argument("--dataset_type", type=str, default='test', help="train/dev/test")
parser.add_argument("--shuffle", type=int, default=1_000, required=False, help="Shuffle buffer size")
parser.add_argument("--max_jz", type=int, default=10, help="Maximum JZ to use")
parser.add_argument("--min_jz", type=int, default=2, help="Maximum JZ to use")
parser.add_argument("--jz_low_cut", action='store_true', help="Cut the dataset")
parser.add_argument("--cut", action='store_true', help="Cut the dataset")
parser.add_argument("--xlim", type=float, nargs=2, default=None, help="X-axis limits")
parser.add_argument("--log_binning", action='store_true', help="Use log-spaced bins")
parser.add_argument("--flatten_jz", action='store_true', help="Flatten each JZ slice individually")
parser.add_argument("--flatten", action='store_true', help="Flatten whole dataset")
parser.add_argument("--equalize_labels", action='store_true', help="Equalize labels in each JZ")
parser.add_argument("--precompute", action='store_true', help="Precompute the initial distribution")
parser.add_argument('-w', "--weight_var", type=str, default=None, help="Use reweighting, specify the weight variable")
parser.add_argument("--min_count", action='store_true', help="Use min count for each bin")

JZ_LOW_PT = [20, 60, 160, 400, 800, 1300, 1800, 2500, 3200, 3900, 4600, 5300]
JZ_LOW_PT = [val * 1e3 for val in JZ_LOW_PT]

JZ_HIGH_PT = [60, 160, 400, 800, 1300, 1800, 2500, 3200, 3900, 4600, 5300, 7000]
JZ_HIGH_PT = [val * 1e3 for val in JZ_HIGH_PT]


@tf.function
def get_labeled_pt(data):
    parton = data['jets_PartonTruthLabelID']
    if tf.equal(parton, tf.constant(21)):
        label = 0
    elif tf.reduce_any(tf.equal(parton, tf.constant([1, 2, 3, 4, 5, 6], dtype=tf.int32))):
        label = 1
    else:
        label = -1
    return {'jets_pt': data['jets_pt'], 'label': label, 'all_label': parton}


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO)
    logging.info(f'Running with args: {{{", ".join([f"{k}: {v}" for k, v in vars(args).items()])}}}')
    os.makedirs(args.save_path, exist_ok=True)

    @tf.function
    def jz_mapper(dataset: tf.data.Dataset, jz: int) -> tf.data.Dataset:
        dataset = dataset.map(write_new_variable(jz))
        if args.jz_low_cut:
            dataset = dataset.filter(get_cut_fn('jets_pt', lower_limit=tf.constant(JZ_LOW_PT)[jz - 1]))
        if args.flatten_jz:
            jz_resampler = partial(resample_var_with_labels,
                                   bins=args.bins,
                                   lower_var_limit=tf.constant(JZ_LOW_PT)[jz - 1],
                                   upper_var_limit=tf.constant(JZ_HIGH_PT)[jz - 1],
                                   log_binning_base=np.e if args.log_binning else None,
                                   variable='jets_pt',
                                   precompute_init_dist=args.precompute,
                                   label_variable='jets_PartonTruthLabelID',
                                   from_min_count=args.min_count)
            dataset = dataset.apply(jz_resampler)
        if args.equalize_labels:
            label_resampler = partial(resample_labels,
                                      label_variable='jets_PartonTruthLabelID',
                                      precompute_init_dist=args.precompute,
                                      from_min_count=args.min_count)
            dataset = dataset.apply(label_resampler)

        return dataset

    files = [os.path.join(args.file_path, f'JZ{jz}', args.dataset_type) for jz in range(args.min_jz, args.max_jz + 1)]
    file_labels = list(range(args.min_jz, args.max_jz + 1))
    dataset = JIDENNDataset.load_parallel(files, dataset_mapper=jz_mapper, file_labels=file_labels)

    dataset = dataset.filter(get_cut_fn('jets_pt', args.xlim[0], args.xlim[1])) if args.cut else dataset

    if args.flatten:
        resampler = partial(resample_var_with_labels,
                            bins=args.bins,
                            lower_var_limit=args.xlim[0] if args.xlim is not None else JZ_LOW_PT[args.min_jz - 1],
                            upper_var_limit=args.xlim[1] if args.xlim is not None else JZ_HIGH_PT[args.max_jz - 1],
                            log_binning_base=np.e if args.log_binning else None,
                            variable='jets_pt',
                            precompute_init_dist=args.precompute,
                            label_variable='jets_PartonTruthLabelID',
                            weight_var=args.weight_var,
                            from_min_count=args.min_count)
        dataset = dataset.apply(resampler)

    dataset = dataset.apply(lambda x: x.shuffle(args.shuffle, seed=42).prefetch(tf.data.AUTOTUNE))
    dataset = dataset.take(args.take) if args.take is not None else dataset
    dataset.save(args.save_path, num_shards=args.num_shards)

    dataset = JIDENNDataset.load(args.save_path)
    print(f'Dataset size: {dataset.length}')


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
