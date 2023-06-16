#!/home/jankovys/JIDENN/venv/bin/python
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
#
from jidenn.preprocessing.plotter import plot_pt_dist
#
logging.basicConfig(level=logging.INFO)
# from src.data.ROOTDataset import ROOTDataset, ROOTVariables
ROOTVariables = dict[str, tf.RaggedTensor]


parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, default='data/pythia_flat_allJZ/train', help="Path to save the dataset")
parser.add_argument("--file_path", type=str, default='data/pythia', help="Path to the root file")
parser.add_argument("--num_shards", type=int, default=256, help="Path to the root file")


def main(args):
    paths = os.listdir(args.file_path)
    paths = [os.path.join(args.file_path, path) for path in paths if 'JZ' in path]
    datasets = [tf.data.Dataset.load(path, compression='GZIP') for path in paths]
    sizes = np.array([dataset.cardinality().numpy() for dataset in datasets])
    sizes = sizes / np.sum(sizes)
    dataset: tf.data.Dataset = tf.data.Dataset.sample_from_datasets(datasets, sizes)
    dataset = dataset.shuffle(100_000, seed=42)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    @tf.function
    def random_shards(x: ROOTVariables) -> tf.Tensor:
        return tf.random.uniform(shape=[], minval=0, maxval=args.num_shards, dtype=tf.int64)

    logging.info(f'Saving dataset to {args.save_path}')
    dataset.save(args.save_path, compression='GZIP', shard_func=random_shards)
    logging.info('Plotting')
    plot_pt_dist(dataset.take(1_000_000), save_path=os.path.join(args.save_path, 'pt_dist.png'))
    logging.info(f'Cardinality: {dataset.cardinality().numpy()}')


if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
