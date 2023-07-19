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


from jidenn.preprocess.dataset_ops import load_dataset, save_dataset
from jidenn.preprocess.resampling import resample_with_labels_dataset, write_new_variable, get_cut_fn, get_filter_fn, resample_dataset
# from jidenn.evaluation.plotter import plot_single_dist


parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, default='data/pythia_flat/train', help="Path to save the dataset")
parser.add_argument("--file_path", type=str, default='data/pythia', help="Path to the root file")
parser.add_argument("--num_shards", type=int, default=256, help="Number of shards to use when saving the dataset")
parser.add_argument("--bins", type=int, default=70, help="Number of bins to use for the binning")
parser.add_argument("--dataset_type", type=str, default='test', help="train/dev/test")
parser.add_argument("--shuffle", type=int, default=10_000, required=False, help="Shuffle buffer size")
parser.add_argument("--max_jz", type=int, default=9, help="Maximum JZ to use")
parser.add_argument("--cut", type=bool, default=True, help="Cut the dataset")
parser.add_argument("--log_binning", action='store_true', help="Use log-spaced bins")
parser.add_argument("--precompute", type=bool, default=True, help="Precompute the initial distribution")


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


    datasets = []
    for jz in range(2, args.max_jz + 1):
        ds = load_dataset(os.path.join(args.file_path, f'JZ{jz}', args.dataset_type))
        ds = ds.map(write_new_variable(jz))
        datasets.append(ds)
    dataset = tf.data.Dataset.sample_from_datasets(datasets, stop_on_empty_dataset=True)
    
    # dataset = dataset.filter(get_cut_fn('jets_pt', 40_000, 4_000_000)) if args.cut else dataset
    dataset = resample_with_labels_dataset(dataset,
                                             bins=args.bins,
                                             lower_var_limit=40_000,
                                             upper_var_limit=4_000_000,
                                             log_binning_base=np.e if args.log_binning else None,
                                             variable='jets_pt',
                                             precompute_init_dist=args.precompute,
                                             label_variable='jets_PartonTruthLabelID',
                                             from_min_count=True)
                                             
    
    # def resample_single(dataset, subdir, ids):
    #     single_dataset = dataset.filter(get_filter_fn('jets_PartonTruthLabelID', ids))
    #     single_dataset = resample_dataset(single_dataset,
    #                                        bins=args.bins,
    #                                        lower_var_limit=60_000,
    #                                        upper_var_limit=3_900_000,
    #                                        log_binning_base=np.e if args.log_binning else None,
    #                                        variable='jets_pt',
    #                                        precompute_init_dist=args.precompute)
        
    #     save_dataset(single_dataset, os.path.join(args.save_path, subdir), args.num_shards // 2)
    
    # os.makedirs(os.path.join(args.save_path, 'tmp/gluon'), exist_ok=True)
    # os.makedirs(os.path.join(args.save_path, 'tmp/quark'), exist_ok=True)
    # resample_single(dataset, 'tmp/gluon', [21])
    # resample_single(dataset, 'tmp/quark', [1, 2, 3, 4, 5, 6])
    # gluon_dataset = load_dataset(os.path.join(args.save_path, 'tmp/gluon'))
    # print(f'Gluon dataset size: {gluon_dataset.cardinality().numpy()}')
    # quark_dataset = load_dataset(os.path.join(args.save_path, 'tmp/quark'))
    # print(f'Quark dataset size: {quark_dataset.cardinality().numpy()}')
    
    # dataset = tf.data.Dataset.sample_from_datasets([gluon_dataset, quark_dataset], [0.5, 0.5], stop_on_empty_dataset=True)
    
    
    dataset = dataset.shuffle(args.shuffle, seed=42) if args.shuffle is not None else dataset
    dataset = dataset.prefetch(tf.data.AUTOTUNE)


    save_dataset(dataset, args.save_path, args.num_shards)
    
    dataset = load_dataset(args.save_path)
    print(f'Dataset size: {dataset.cardinality().numpy()}')
    
    # remove tmp dir and everythiong in it
    # shutil.rmtree(os.path.join(args.save_path, 'tmp'))



if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
