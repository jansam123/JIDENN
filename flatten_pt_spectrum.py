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
logging.basicConfig(level=logging.INFO)
# from src.data.ROOTDataset import ROOTDataset, ROOTVariables
ROOTVariables = dict[str, tf.RaggedTensor]

parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, default='data/pythia_flat_allJZ/train', help="Path to save the dataset")
parser.add_argument("--file_path", type=str, default='data/pythia', help="Path to the root file")
parser.add_argument("--num_shards", type=int, default=256, help="Path to the root file")
parser.add_argument("--bins", type=int, default=100, help="Path to the root file")
parser.add_argument("--dataset_type", type=str, default='train', help="train/dev/test")

# CPUS = tf.config.experimental.list_physical_devices('CPU')
# logging.info(f'CPUS: {CPUS}')
# logging.info(f'Num of threads: {tf.config.threading.get_inter_op_parallelism_threads()}')
# tf.config.threading.set_inter_op_parallelism_threads(0)
# tf.config.threading.set_intra_op_parallelism_threads(0)
# logging.info(f'Num of threads: {tf.config.threading.get_inter_op_parallelism_threads()}')


def write_JZ_wrapper(jz_slice: int) -> Callable[[ROOTVariables], ROOTVariables]:
    @tf.function
    def write_jz(data: ROOTVariables) -> ROOTVariables:
        new_data = data.copy()
        new_data['JZ_slice'] = jz_slice
        return new_data
    return write_jz


def rebin_wrapper(n_bins, pt_range: List[float] = [60_000., 5_600_000.]) -> Callable[[ROOTVariables], tf.Tensor]:
    @tf.function
    def rebin_pt(data: ROOTVariables) -> tf.Tensor:
        jet_pt = data['jets_pt']
        jet_pt = tf.reshape(jet_pt, ())
        index = tf.histogram_fixed_width_bins(jet_pt, pt_range, nbins=n_bins + 1)
        index = index - 1
        if index < 0:
            index = tf.constant(0, dtype=tf.int32)
        return index
        # parton = data['jets_PartonTruthLabelID']
        # if tf.equal(parton, tf.constant(21)):
        #     return index
        # elif tf.reduce_any(tf.equal(parton, tf.constant([1, 2, 3, 4, 5, 6], dtype=tf.int32))):
        #     return index + n_bins
        # else:
        #     return tf.constant(0, dtype=tf.int32)
    return rebin_pt


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

    dataset = dataset.map(get_labeled_pt)
    df = tfds.as_dataframe(dataset)
    df['Truth Label'] = df['label'].replace({1: 'quark', 0: 'gluon'})
    df['All Truth Label'] = df['all_label'].replace({1: 'd', 2: 'u', 3: 's', 4: 'c', 5: 'b', 6: 't', 21: 'g'})
    sns.histplot(data=df, x='jets_pt', hue='Truth Label',
                 stat='count', element="step", fill=True,
                 palette='Set1', common_norm=False, hue_order=['quark', 'gluon'])
    plt.savefig(save_path)
    plt.yscale('log')
    plt.savefig(save_path.replace('.png', '_log.png'))
    plt.close()

    sns.histplot(data=df, x='jets_pt', hue='All Truth Label',
                 stat='count', element="step", fill=True, multiple='stack',
                 palette='Set1', common_norm=False)
    plt.savefig(save_path.replace('.png', '_all.png'))
    plt.yscale('log')
    plt.savefig(save_path.replace('.png', '_all_log.png'))


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.save_path, exist_ok=True)

    @tf.function
    def split_q_g(data):

        parton = data['jets_PartonTruthLabelID']
        if tf.equal(parton, tf.constant(21)):
            label = 0
        elif tf.reduce_any(tf.equal(parton, tf.constant([1, 2, 3, 4, 5, 6], dtype=tf.int32))):
            label = 1
        else:
            label = 0
        return label

    JZ_slices = ['JZ02_r10724',
                 'JZ03_r10724',
                 'JZ04_r10724',
                 'JZ05_r10724',
                 'JZ06_r10724',
                 'JZ07_r10724',
                 'JZ08_r10724',
                 'JZ09_r10724',
                 'JZ10_r10724',
                 'JZ11_r10724',
                 'JZ12_r10724',
                 ]

    pt_cuts = [
        60_000.,
        160_000.,
        400_000.,
        800_000.,
        1_300_000.,
        1_800_000.,
        2_500_000.,
        3_200_000.,
        3_900_000.,
        4_600_000.,
        5_300_000.,
    ]

    def up_cut_pt_wrapper(pt):
        @tf.function
        def ucut_pt(data):
            return tf.reduce_all(data['jets_pt'] > pt)
        return ucut_pt

    def down_cut_pt_wrapper(pt):
        @tf.function
        def dcut_pt(data):
            return tf.reduce_all(data['jets_pt'] < pt)
        return dcut_pt

    def double_cut_pt_wrapper(pt_high, pt_low):
        @tf.function
        def double_cut_pt(data):
            return tf.reduce_all([data['jets_pt'] > pt_low, data['jets_pt'] < pt_high])
        return double_cut_pt

    datasets = []
    sizes = []
    quarks = []
    gluons = []
    for i, jz_slice in enumerate(JZ_slices):
        file = f'{args.file_path}/{jz_slice}/{args.dataset_type}'
        with open(os.path.join(file, 'element_spec'), 'rb') as f:
            element_spec = pickle.load(f)
        ds = tf.data.Dataset.load(file, compression='GZIP')
        sizes += [ds.cardinality().numpy()]
        ds = ds.filter(up_cut_pt_wrapper(pt_cuts[i]))
        # ds = ds.rejection_resample(split_q_g, target_dist=[0.5, 0.5]).map(lambda x, y: y)
        # pt_ranges = [pt_cuts[i], pt_cuts[i + 1]] if i < len(pt_cuts) - 1 else [pt_cuts[i], 5_800_000]
        # ds = ds.rejection_resample(rebin_wrapper(args.bins, pt_ranges), target_dist=[
        #                            1 / (args.bins * 2)] * args.bins * 2, seed=42).map(lambda w, z: z)
        ds = ds.map(write_JZ_wrapper(i + 1))
        if i > 5:
            ds = ds.repeat()
        gluons.append(ds.filter(lambda x: tf.equal(x['jets_PartonTruthLabelID'], tf.constant(21))))
        quarks.append(ds.filter(lambda x: tf.reduce_any(
            tf.equal(x['jets_PartonTruthLabelID'], tf.constant([1, 2, 3, 4, 5, 6], dtype=tf.int32)))))

        datasets.append(ds)

    sizes = tf.constant(sizes, dtype=tf.float32)
    sizes = sizes / tf.reduce_sum(sizes)
    # dataset: tf.data.Dataset = tf.data.Dataset.sample_from_datasets(datasets, sizes)
    gluons = tf.data.Dataset.sample_from_datasets(gluons, sizes[::-1])
    gluons = gluons.rejection_resample(rebin_wrapper(args.bins), target_dist=[
        1 / (args.bins)] * args.bins, seed=42).map(lambda w, z: z)
    quarks = tf.data.Dataset.sample_from_datasets(quarks, sizes[::-1])
    quarks = quarks.rejection_resample(rebin_wrapper(args.bins), target_dist=[
        1 / (args.bins)] * args.bins, seed=42).map(lambda w, z: z)
    dataset = tf.data.Dataset.sample_from_datasets([gluons, quarks], [0.5, 0.5], stop_on_empty_dataset=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # logging.info(f'Plotting pt spectrum before resampling.')
    # plot_pt_dist(dataset.take(1_000_000), save_path=f'{args.save_path}/pt_spectrum.png')
    logging.info(f'Resampling dataset.')

    # bins = np.logspace(np.log10(60_000), np.log10(5_600_000), args.bins)
    # all_dataset = []
    # for i in range(args.bins + 1):
    #     if i == 0:
    #         ds = dataset.filter(down_cut_pt_wrapper(bins[i]))
    #     elif i == args.bins:
    #         ds = dataset.filter(up_cut_pt_wrapper(bins[i - 1]))
    #     else:
    #         ds = dataset.filter(double_cut_pt_wrapper(bins[i], bins[i - 1]))
    #     ds = ds.rejection_resample(split_q_g, target_dist=[0.5, 0.5]).map(lambda x, y: y)
    #     all_dataset.append(ds)

    # bins = np.concatenate([np.array([30_000]), bins])
    # dataset = tf.data.Dataset.sample_from_datasets(all_dataset, (bins / np.sum(bins))[::-1])
    # dataset = dataset.shuffle(100_000, seed=42)
    # dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # target_dist = [1 / (args.bins * 2)] * args.bins * 2
    # dataset = dataset.rejection_resample(
    #     rebin_wrapper(args.bins), target_dist=target_dist, seed=42).map(lambda w, z: z)

    with open(os.path.join(args.save_path, 'element_spec'), 'wb') as f:
        pickle.dump(dataset.element_spec, f)

    @tf.function
    def random_shards(x: ROOTVariables) -> tf.Tensor:
        return tf.random.uniform(shape=[], minval=0, maxval=args.num_shards, dtype=tf.int64)

    dataset.save(path=args.save_path, compression='GZIP',
                 shard_func=random_shards)
    dataset = tf.data.Dataset.load(args.save_path, compression='GZIP')
    cardinality = dataset.cardinality().numpy()
    logging.info(cardinality)
    take = 1_000_000 if cardinality > 1_000_000 else cardinality
    logging.info(f'Plotting pt spectrum after resampling.')
    plot_pt_dist(dataset.take(take), save_path=f'{args.save_path}/pt_spectrum_rebinned.png')
    logging.info(f'Done.')


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
    # ds = tf.data.Dataset.load('data/pythia_allJZ_flat/test', compression='GZIP')
    # ds = ds.shuffle(100_000, seed=42)
    # plot_pt_dist(ds, save_path=f'data/pythia_allJZ_flat/test/pt_spectrum_rebinned_shuffled.png')
