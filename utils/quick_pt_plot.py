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
logging.basicConfig(level=logging.INFO)
# from src.data.ROOTDataset import ROOTDataset, ROOTVariables
ROOTVariables = dict[str, tf.RaggedTensor]

parser = argparse.ArgumentParser()
parser.add_argument("--plot_path", type=str,
                    default='/home/jankovys/JIDENN/data/pythia_allJZ_flat/test', help="Path where to save the plots")
parser.add_argument("--dataset_path", type=str,
                    default='/home/jankovys/JIDENN/data/pythia_allJZ_flat/test', help="Path to the dataset")


def plot_pt_dist(dataset: tf.data.Dataset, save_path: str = 'figs.png') -> None:

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

    # log_bins = np.logspace(np.log10(6e4), np.log10(6e6), 50)
    
    dataset = dataset.map(get_labeled_pt)
    df = tfds.as_dataframe(dataset)
    df['Truth Label'] = df['label'].replace({1: 'quark', 0: 'gluon'})
    df['All Truth Label'] = df['all_label'].replace({1: 'd', 2: 'u', 3: 's', 4: 'c', 5: 'b', 6: 't', 21: 'g'})
    sns.histplot(data=df, x='jets_pt', hue='Truth Label',
                 stat='count', element="step", fill=True,
                 palette='Set1', common_norm=False, hue_order=['quark', 'gluon'], bins=50)
    plt.savefig(save_path)
    plt.yscale('log')
    plt.savefig(save_path.replace('.png', '_log.png'))
    plt.close()

    sns.histplot(data=df, x='jets_pt', hue='All Truth Label',
                 stat='count', element="step", fill=True, multiple='stack',
                 palette='Set1', common_norm=False, bins=50)
    plt.savefig(save_path.replace('.png', '_all.png'))
    plt.yscale('log')
    plt.savefig(save_path.replace('.png', '_all_log.png'))
    plt.close()


def main(args: argparse.Namespace) -> None:
    dataset = tf.data.Dataset.load(args.dataset_path, compression='GZIP')
    logging.info(f"Loaded dataset from {args.dataset_path}")
    cardinality = dataset.cardinality().numpy()
    logging.info(f"Dataset cardinality: {cardinality}")
    take = 1_000_000 if cardinality > 1_000_000 else cardinality
    plot_pt_dist(dataset.take(50_000), save_path=os.path.join(args.plot_path, 'pt_dist.png'))


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
