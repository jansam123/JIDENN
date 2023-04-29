#!/home/jankovys/JIDENN/venv/bin/python
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import argparse
import pickle
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple, Union
#
logging.basicConfig(level=logging.INFO)
# from src.data.ROOTDataset import ROOTDataset, ROOTVariables
ROOTVariables = dict[str, tf.RaggedTensor]

parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, default='data/pythia_flat/train', help="Path to save the dataset")
parser.add_argument("--file_path", type=str, default='data/pythia', help="Path to the root file")
parser.add_argument("--plot", type=int, default=0, help="Plot the dataset")
parser.add_argument("--num_shards", type=int, default=256, help="Path to the root file")
parser.add_argument("--dataset_type", type=str, default='train', help="train/dev/test")

tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)


def write_JZ_wrapper(jz_slice: int) -> Callable[[ROOTVariables], ROOTVariables]:
    @tf.function
    def write_jz(data: ROOTVariables) -> ROOTVariables:
        new_data = data.copy()
        new_data['JZ_slice'] = jz_slice
        return new_data
    return write_jz


PT_BINS = 10
ETA_BINS = 4


@tf.function
def rebin_pt(data: ROOTVariables) -> tf.Tensor:
    pt_range = [30_000., 1_000_000.]
    # eta_bins = [0.0, 0.6, 1.2, 1.8, 2.4]
    eta_range = [0.6, 2.0]
    jet_pt = data['jets_pt']
    jets_eta = data['jets_eta']
    jets_eta = tf.abs(jets_eta)
    jets_eta = tf.reshape(jets_eta, ())
    jet_pt = tf.reshape(jet_pt, ())
    index = tf.histogram_fixed_width_bins(jet_pt, pt_range, nbins=PT_BINS)
    eta_index = tf.histogram_fixed_width_bins(jets_eta, eta_range, nbins=ETA_BINS)
    parton = data['jets_PartonTruthLabelID']
    id_index = 0 if tf.equal(parton, tf.constant(21)) else 1
    multi_index = tf.stack([index, eta_index, id_index])
    print(multi_index)
    dims = tf.constant([PT_BINS, ETA_BINS, 2])
    strides = tf.math.cumprod(dims, exclusive=True, reverse=True)
    return tf.reduce_sum(multi_index * tf.expand_dims(strides, 1), axis=0)

    # if tf.equal(parton, tf.constant(21)):
    #     return index + PT_BINS * eta_index
    # elif tf.reduce_any(tf.equal(parton, tf.constant([1, 2, 3, 4, 5, 6], dtype=tf.int32))):
    #     return index + PT_BINS * eta_index + PT_BINS * ETA_BINS
    # else:
    #     return tf.constant(0)


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.save_path, exist_ok=True)

    JZ_slices = ['JZ01_r10724',
                 'JZ02_r10724',
                 'JZ03_r10724',
                 'JZ04_r10724',
                 'JZ05_r10724', ]

    datasets = []
    for i, jz_slice in enumerate(JZ_slices):
        file = f'{args.file_path}/{jz_slice}/{args.dataset_type}'
        with open(os.path.join(file, 'element_spec'), 'rb') as f:
            element_spec = pickle.load(f)
        ds = tf.data.Dataset.load(file, compression='GZIP')
        ds = ds.map(write_JZ_wrapper(i + 1), num_parallel_calls=tf.data.AUTOTUNE)
        datasets.append(ds)

    dataset: tf.data.Dataset = tf.data.Dataset.sample_from_datasets(datasets, stop_on_empty_dataset=True)
    target_dist = [1 / (PT_BINS * ETA_BINS * 2)] * PT_BINS * ETA_BINS * 2
    dataset = dataset.rejection_resample(
        rebin_pt, target_dist=target_dist, seed=42).map(lambda x, data: data)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    with open(os.path.join(args.save_path, 'element_spec'), 'wb') as f:
        pickle.dump(dataset.element_spec, f)

    @tf.function
    def random_shards(x: ROOTVariables) -> tf.Tensor:
        return tf.random.uniform(shape=[], minval=0, maxval=args.num_shards, dtype=tf.int64)

    dataset.save(path=args.save_path, compression='GZIP',
                 shard_func=random_shards)


def plot(args):
    dataset = tf.data.Dataset.load(args.save_path, compression='GZIP')
    print(dataset.cardinality())
    if args.plot == 0:
        dataset = dataset.take(100_000)

    @tf.function
    def get_labeled_pt(data):
        parton = data['jets_PartonTruthLabelID']
        if tf.equal(parton, tf.constant(21)):
            label = 0
        elif tf.reduce_any(tf.equal(parton, tf.constant([1, 2, 3, 4, 5, 6], dtype=tf.int32))):
            label = 1
        else:
            label = -1
        return {'jets_pt': data['jets_pt'], 'label': label}

    dataset = dataset.map(get_labeled_pt)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    df = tfds.as_dataframe(dataset)
    print(df['label'].unique())
    df['Truth Label'] = df['label'].replace({1: 'quark', 0: 'gluon'})
    print(df)
    sns.histplot(data=df, x='jets_pt', hue='Truth Label',
                 stat='count', element="step", fill=True,
                 palette='Set1', common_norm=False, hue_order=['quark', 'gluon'])
    # plt.yscale('log')
    plt.savefig(f'{args.save_path}/pt_spectrum.png')

    sns.histplot(data=df, x='jets_eta', hue='Truth Label',
                 stat='count', element="step", fill=True,
                 palette='Set1', common_norm=False, hue_order=['quark', 'gluon'])
    plt.savefig(f'{args.save_path}/eta_spectrum.png')


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    if args.plot == 0:
        main(args)
    plot(args)
