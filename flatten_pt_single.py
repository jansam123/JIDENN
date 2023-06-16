#!/home/jankovys/JIDENN/venv/bin/python
import tensorflow as tf
import os
import argparse
import pickle
import logging
#
from jidenn.preprocessing.flattening_fns import (
    get_bin_fn,
    get_label_bin_fn,
    write_new_variable,
    get_label_splitting_fn,
    get_cut_fn,
    get_filter_fn
)
from jidenn.preprocessing.plotter import plot_pt_dist

logging.basicConfig(level=logging.INFO)
ROOTVariables = dict[str, tf.RaggedTensor]

parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, default='data/pythia_flat_allJZ/train', help="Path to save the dataset")
parser.add_argument("--file_path", type=str, default='data/pythia', help="Path to the root file")
parser.add_argument("--num_shards", type=int, default=256, help="Path to the root file")
parser.add_argument("--jz_slice", type=int, default=1, help="JZ slice to process, 1-12")
parser.add_argument("--bins", type=int, default=100, help="Path to the root file")
parser.add_argument("--dataset_type", type=str, default='train', help="train/dev/test")


JZ_SLICES = [
    'JZ1',
    'JZ2',
    'JZ3',
    'JZ4',
    'JZ5',
    'JZ6',
    'JZ7',
    'JZ8',
    'JZ9',
    'JZ10',
    'JZ11',
    'JZ12',
]

PT_BINS = [
    20_000.,
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


def main(args: argparse.Namespace) -> None:
    jz_slice = JZ_SLICES[args.jz_slice - 1]
    args.save_path = os.path.join(args.save_path, jz_slice)
    os.makedirs(args.save_path, exist_ok=True)

    file = f'{args.file_path}/{jz_slice}/{args.dataset_type}'

    dataset = tf.data.Dataset.load(file, compression='GZIP')
    if args.jz_slice < len(JZ_SLICES):
        pt_range = (PT_BINS[args.jz_slice - 1], PT_BINS[args.jz_slice])
    else:
        pt_range = (PT_BINS[args.jz_slice - 1], 5_800_000)

    plot_pt_dist(dataset.take(50_000).prefetch(tf.data.AUTOTUNE),
                 save_path=f'{args.save_path}/pt_spectrum.png')

    dataset = dataset.filter(get_cut_fn(lower_limit=pt_range[0], upper_limit=pt_range[1]))

    gluons = dataset.filter(get_filter_fn(variable='jets_PartonTruthLabelID', values=[21]))
    gluons = gluons.rejection_resample(get_bin_fn(n_bins=args.bins, lower_pt_limit=pt_range[0], upper_pt_limit=pt_range[1]), target_dist=[
                                       1 / (args.bins)] * args.bins, seed=42).map(lambda w, z: z)

    quarks = dataset.filter(get_filter_fn(variable='jets_PartonTruthLabelID', values=[1, 2, 3, 4, 5, 6]))
    quarks = quarks.rejection_resample(get_bin_fn(n_bins=args.bins, lower_pt_limit=pt_range[0], upper_pt_limit=pt_range[1]), target_dist=[
                                       1 / (args.bins)] * args.bins, seed=42).map(lambda w, z: z)

    dataset = tf.data.Dataset.sample_from_datasets([gluons, quarks], [0.5, 0.5], stop_on_empty_dataset=True)
    dataset = dataset.map(write_new_variable(variable_value=args.jz_slice, variable_name='JZ_slice'))

    with open(os.path.join(args.save_path, 'element_spec'), 'wb') as f:
        pickle.dump(dataset.element_spec, f)

    @tf.function
    def random_shards(x: ROOTVariables) -> tf.Tensor:
        return tf.random.uniform(shape=[], minval=0, maxval=args.num_shards, dtype=tf.int64)

    logging.info(f"Saving dataset to {args.save_path}")
    dataset.save(path=args.save_path, compression='GZIP',
                 shard_func=random_shards)
    logging.info(f"Saved dataset to {args.save_path}")
    plot_pt_dist(dataset.take(50_000).prefetch(tf.data.AUTOTUNE),
                 save_path=f'{args.save_path}/pt_spectrum_rebinned.png')
    logging.info('Done')


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
