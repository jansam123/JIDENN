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


from jidenn.preprocess.dataset_ops import load_dataset, save_dataset
from jidenn.preprocess.resampling import resample_with_labels_dataset, write_new_variable, get_cut_fn, get_filter_fn, resample_dataset
from jidenn.evaluation.plotter import plot_single_dist
from plot_distribution import plot_single


parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, default='data/pythia_flat/train', help="Path to save the dataset")
parser.add_argument("--file_path", type=str, default='data/pythia', help="Path to the root file")
parser.add_argument("--num_shards", type=int, default=256, help="Number of shards to use when saving the dataset")


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


def write_weights(cross_section: float = 1., filt_eff: float = 1., additional_weight: float = 1.):
    @tf.function
    def _calculate_weights(data):
        norm = data['metadata'][1]
        weight = data['weight_mc'][0] * additional_weight * cross_section * filt_eff / norm
        return {**data, 'weight': weight}
    return _calculate_weights


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO)
    logging.info(f'Running with args: {{{", ".join([f"{k}: {v}" for k, v in vars(args).items()])}}}')
    os.makedirs(args.save_path, exist_ok=True)
    jz_description = pd.read_csv(os.path.join('data', 'JZ_description.csv'))

    datasets = []
    for jz in range(2, 13):
        ds = load_dataset(os.path.join(args.file_path, f'JZ{jz}', 'test'))
        ds = ds.map(write_new_variable(jz))
        filt_eff = jz_description.iloc[jz]['filtEff']
        cross_section = jz_description.iloc[jz]['crossSection [pb]']
        ds = ds.filter(lambda x: tf.rank(x['metadata']) > 0)
        ds = ds.map(write_weights(cross_section, filt_eff))
        datasets.append(ds)
    dataset = tf.data.Dataset.sample_from_datasets(datasets, stop_on_empty_dataset=False)
    # dataset = dataset.take(1_000_000)
    # dataset = dataset.map(get_labeled_pt)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # for i in dataset.take(100):
    #     print(i['metadata'].numpy(), i['eventNumber'].numpy()[0])

    save_dataset(dataset, args.save_path, args.num_shards)

    dataset = load_dataset(args.save_path)
    print(f'Dataset size: {dataset.cardinality().numpy()}')

    plot_single(dataset, variable='jets_pt', hue_var='jets_PartonTruthLabelID', bins=70, xlim=(0.02, 7),
                badge_text='Pythia 8', save_path='pythia_pt_dist.png', weight_var='weight', ylog=True,
                log_bins=False, ylim=(1e1, 1e8))


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
