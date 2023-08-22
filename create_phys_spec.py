import os
import argparse
import logging
from functools import partial
import tensorflow as tf
import pandas as pd
import numpy as np
logging.basicConfig(format='[%(asctime)s][%(levelname)s] - %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
#

from jidenn.data.JIDENNDataset import JIDENNDataset
from jidenn.preprocess.resampling import get_cut_fn
from jidenn.preprocess.resampling import write_new_variable


parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, help="Path to save the dataset")
parser.add_argument("--load_path", type=str, help="Path to the root file")
parser.add_argument("--shuflle", type=int, default=100_000, required=False, help="Shuffle buffer size")
parser.add_argument("--num_shards", type=int, default=256, required=False,
                    help="Number of shards to save the dataset in")
parser.add_argument("--min_jz", type=int, default=2, help="Maximum JZ to use")
parser.add_argument("--max_jz", type=int, default=10, help="Maximum JZ to use")
parser.add_argument("--dataset_type", type=str, default='test', help="train/dev/test")


JZ_LOW_PT = [20, 60, 160, 400, 800, 1300, 1800, 2500, 3200, 3900, 4600, 5300]
JZ_LOW_PT = [val * 1e3 for val in JZ_LOW_PT]

JZ_HIGH_PT = [60, 160, 400, 800, 1300, 1800, 2500, 3200, 3900, 4600, 5300, 7000]
JZ_HIGH_PT = [val * 1e3 for val in JZ_HIGH_PT]

MAX_FRAC = 1.4
TAKE_FRAC = 0.01


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.save_path, exist_ok=True)
    files = [os.path.join(args.load_path, f'JZ{jz}', args.dataset_type) for jz in range(args.min_jz, args.max_jz + 1)]
    sizes = [tf.data.Dataset.load(file, compression='GZIP').cardinality().numpy() for file in files]
    sampling_weights = [size / sum(sizes) for size in sizes]
    print(sampling_weights)

    @tf.function
    def jz_cutter(dataset: tf.data.Dataset, jz) -> tf.data.Dataset:
        dataset = dataset.filter(get_cut_fn('jets_pt', upper_limit=MAX_FRAC * tf.constant(JZ_HIGH_PT)[jz - 1]))
        dataset = dataset.filter(write_new_variable('JZ_slice', jz))
        return dataset

    file_labels = list(range(args.min_jz, args.max_jz + 1))
    # dataset = JIDENNDataset.load_parallel(files, dataset_mapper=jz_cutter, file_labels=file_labels)
    dataset = JIDENNDataset.load_multiple(files, dataset_mapper=jz_cutter,
                                          file_labels=file_labels, weights=sampling_weights)
    # dataset = dataset.filter(lambda x: tf.random.uniform([]) < TAKE_FRAC)
    events = dataset.to_numpy(['eventNumber'])
    print(f"Number of events: {events['eventNumber'].shape[0]}")
    event_numbers = np.unique(events['eventNumber'])
    total_events = event_numbers.shape[0]
    print(f'Total number of unique events: {total_events}')
    selected_number_of_events = int(total_events * TAKE_FRAC)
    print(f'Selected number of events: {selected_number_of_events}')
    event_numbers = tf.constant(event_numbers, dtype=tf.int32)
    selected_events = tf.random.shuffle(event_numbers)[:selected_number_of_events]
    dataset = dataset.filter(lambda x: tf.reduce_any(tf.equal(x['eventNumber'], selected_events)))
    dataset = dataset.apply(lambda x: x.shuffle(600_000))
    dataset = dataset.apply(lambda x: x.prefetch(tf.data.AUTOTUNE))
    dataset.save(args.save_path, num_shards=args.num_shards)
    logging.info(f'Saved dataset to {args.save_path}')
    dataset = JIDENNDataset.load(args.save_path)
    size = dataset.length
    print(f"Number of jets: {size}")
    dataset.plot_single_variable('jets_pt', os.path.join(args.save_path, 'pt.png'), bins=100, ylim=(1e-4, 1e2),
                                 weight_variable='weight', multiple='stack',
                                 ylog=True, hue_variable='JZ_slice', xlabel=r'$p_\mathrm{T}$ [TeV]')

    dataset.plot_single_variable('jets_pt', os.path.join(args.save_path, 'pt_noW.png'), bins=100,
                                 badge_text=f'N = {size:,}', badge=True, multiple='stack',
                                 ylog=True, hue_variable='JZ_slice', xlabel=r'$p_\mathrm{T}$ [TeV]')

    dataset.plot_single_variable('weight', os.path.join(args.save_path, 'weight.png'), bins=100,
                                 badge=False, multiple='stack', stat='count',
                                 ylog=True, hue_variable='JZ_slice', xlabel=r'weight')


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
