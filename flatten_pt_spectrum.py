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
parser.add_argument("--save_path", type=str, default='data/dataset2_flat_b100/train', help="Path to save the dataset")
parser.add_argument("--file_path", type=str, default='data/dataset2_3', help="Path to the root file")
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


@tf.function
def rebin_pt(data: ROOTVariables) -> tf.Tensor:
    nbins = 100
    value_range = [20_000., 1_100_000.]
    new_values = data['jets_pt']
    new_values = tf.reshape(new_values, ())
    index = tf.histogram_fixed_width_bins(new_values, value_range, nbins=nbins)
    parton = data['jets_PartonTruthLabelID']
    if tf.equal(parton, tf.constant(21)):
        return index
    elif tf.reduce_any(tf.equal(parton, tf.constant([1, 2, 3, 4, 5, 6], dtype=tf.int32))):
        return index + nbins
    else:
        return tf.constant(0)


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
        ds = tf.data.experimental.load(file, compression='GZIP')
        ds = ds.map(write_JZ_wrapper(i + 1), num_parallel_calls=tf.data.AUTOTUNE)
        datasets.append(ds)

    dataset: tf.data.Dataset = tf.data.Dataset.sample_from_datasets(datasets, stop_on_empty_dataset=True)
    dataset = dataset.rejection_resample(
        rebin_pt, target_dist=[1 / (100 * 2)] * 100 * 2, seed=42).map(lambda x, data: data)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    with open(os.path.join(args.save_path, 'element_spec'), 'wb') as f:
        pickle.dump(dataset.element_spec, f)

    @tf.function
    def random_shards(x: ROOTVariables) -> tf.Tensor:
        return tf.random.uniform(shape=[], minval=0, maxval=args.num_shards, dtype=tf.int64)

    # os.makedirs(f'{args.save_path}/checkpoints', exist_ok=True)
    # checkpoint_prefix = f'{args.save_path}/checkpoints'
    # step_counter = tf.Variable(0, trainable=False)
    # checkpoint_args = {
    #     "checkpoint_interval": 50,
    #     "step_counter": step_counter,
    #     "directory": checkpoint_prefix,
    #     "max_to_keep": 20,
    # }
    tf.data.experimental.save(dataset, path=args.save_path, compression='GZIP',
                              shard_func=random_shards)  # , checkpoint_args=checkpoint_args)

    dataset = tf.data.experimental.load(args.save_path, compression='GZIP')
    print(dataset.cardinality())
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


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
