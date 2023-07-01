import tensorflow as tf
import tensorflow_datasets as tfds
import os
import argparse
import pickle
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, List, Tuple, Union


from jidenn.preprocess.dataset_ops import load_dataset, save_dataset
from jidenn.preprocess.resampling import resample_with_labels_dataset, write_new_variable
from jidenn.evaluation.plotter import plot_single_dist


parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, default='data/pythia_flat/train', help="Path to save the dataset")
parser.add_argument("--file_path", type=str, default='data/pythia', help="Path to the root file")
parser.add_argument("--num_shards", type=int, default=256, help="Number of shards to use when saving the dataset")
parser.add_argument("--bins", type=int, default=50, help="Number of bins to use for the binning")
parser.add_argument("--dataset_type", type=str, default='test', help="train/dev/test")
parser.add_argument("--shuflle", type=int, default=10_000, required=False, help="Shuffle buffer size")
parser.add_argument("--take", type=int, default=100_000, required=False,
                    help="Number of samples to take from the dataset")


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


def plot_pt_dist(dataset, save_path, take):
    dataset = dataset.take(take)
    dataset = dataset.map(get_labeled_pt)
    df = tfds.as_dataframe(dataset)
    df['Truth Label'] = df['label'].replace({1: 'quark', 0: 'gluon'})
    df['All Truth Label'] = df['all_label'].replace({1: 'd', 2: 'u', 3: 's', 4: 'c', 5: 'b', 6: 't', 21: 'g'})

    plot_single_dist(df, variable='jets_pt', hue_var='Truth Label', hue_order=[
                     'quark', 'gluon'], save_path=f'{save_path}/pt_spec.png')
    plot_single_dist(df, variable='jets_pt', hue_var='All Truth Label', save_path=f'{save_path}/all_pt_spec.png')

    plot_single_dist(df, variable='jets_pt', hue_var='Truth Label', hue_order=[
                     'quark', 'gluon'], save_path=f'{save_path}/pt_spec_log.png', ylog=True)
    plot_single_dist(df, variable='jets_pt', hue_var='All Truth Label',
                     save_path=f'{save_path}/all_pt_spec_log.png', ylog=True)


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.save_path, exist_ok=True)
    # file_paths = [f'{args.file_path}/JZ{i}/{args.dataset_type}' for i in range(2, 10)]

    # element_spec_path = os.path.join(file_paths[0], 'element_spec')
    # with open(element_spec_path, 'rb') as f:
    #     element_spec = pickle.load(f)

    # @tf.function
    # def recast(data):
    #     data = data.copy()
    #     data = {var: tf.cast(data[var], element_spec[var].dtype) for var in element_spec.keys()}
    #     return data

    # @tf.function
    # def load_ds(file_path):
    #     # .map(write_new_variable(i))
    #     return tf.data.Dataset.load(file_path, compression='GZIP', element_spec=element_spec)

    # dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    # dataset = dataset.enumerate(start=2)
    # dataset = dataset.interleave(map_func=load_ds)

    datasets = []
    for jz in range(2, 10):
        ds = load_dataset(os.path.join(args.file_path, f'JZ{jz}', args.dataset_type))
        ds = ds.map(write_new_variable(jz))
        datasets.append(ds)
    dataset = tf.data.Dataset.sample_from_datasets(datasets)

    try:
        os.makedirs(f'{args.save_path}/plots_before', exist_ok=True)
        plot_pt_dist(dataset, f'{args.save_path}/plots_before', take=args.take)
    except Exception as e:
        print(e)

    for i in dataset.take(1):
        print(i)

    dataset = resample_with_labels_dataset(dataset,
                                           n_bins=args.bins,
                                           lower_var_limit=60_000,
                                           upper_var_limit=3_900_000,
                                           variable='jets_pt',
                                           label_variable='jets_PartonTruthLabelID',
                                           label_class_1=[1, 2, 3, 4, 5, 6],
                                           label_class_2=[21],)

    dataset = dataset.shuffle(args.shuflle, seed=42) if args.shuflle is not None else dataset
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    for i in dataset.take(1):
        print(i)

    @tf.function
    def gen_random_number(sample) -> tf.Tensor:
        return tf.random.uniform(shape=[], minval=0, maxval=256, dtype=tf.int64)

    dataset.save(args.save_path, compression='GZIP', shard_func=gen_random_number)

    try:
        os.makedirs(f'{args.save_path}/plots_after', exist_ok=True)
        plot_pt_dist(dataset, f'{args.save_path}/plots_after', take=args.take)
    except Exception as e:
        print(e)
        dataset = load_dataset(args.save_path)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
