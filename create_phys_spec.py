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
from jidenn.preprocess.flatten_dataset import flatten_dataset


parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, help="Path to save the dataset")
parser.add_argument("--load_path", type=str, help="Path to the root file")
parser.add_argument("--shuffle", type=int, default=100_000, required=False, help="Shuffle buffer size")
parser.add_argument("--take", type=int, default=200_000, help="Number of jets to take")
parser.add_argument("--num_shards", type=int, default=256, required=False,
                    help="Number of shards to save the dataset in")
parser.add_argument("--min_jz", type=int, default=2, help="Maximum JZ to use")
parser.add_argument("--max_jz", type=int, default=7, help="Maximum JZ to use")
parser.add_argument("--eta_cut", type=float, default=2.1, help="Eta cut")
parser.add_argument("--dataset_type", type=str, default='test', help="train/dev/test")
parser.add_argument("--reference_variable", type=str, default='jets_PartonTruthLabelID',
                    help="Variable to use as reference for flattening")
parser.add_argument("--wanted_values", type=int, nargs='+',
                    default=[1, 2, 3, 4, 5, 6, 21], help="Values to keep in the reference variable")
parser.add_argument("--jz_description_file", type=str,
                    default='data/JZ_description.csv', help="JZ description csv file.")


JZ_LOW_PT = [20, 60, 160, 400, 800, 1300, 1800, 2500, 3200, 3900, 4600, 5300]
JZ_LOW_PT = [val * 1e3 for val in JZ_LOW_PT]

JZ_HIGH_PT = [60, 160, 400, 800, 1300, 1800, 2500, 3200, 3900, 4600, 5300, 7000]
JZ_HIGH_PT = [val * 1e3 for val in JZ_HIGH_PT]

MAX_FRAC = 1.4
TAKE_FRAC = 0.01


def write_weights(cross_section: float = 1., filt_eff: float = 1., lumi: float = 1., norm: float = 1.):
    @tf.function
    def _calculate_weights(data):
        w = tf.cast(data['weight_mc'], tf.float64)
        cast_lumi = tf.cast(lumi, tf.float64)
        cast_cross_section = tf.cast(cross_section, tf.float64)
        cast_filt_eff = tf.cast(filt_eff, tf.float64)
        cast_norm = tf.cast(norm, tf.float64)
        weight = w[0] * cast_lumi * cast_cross_section * cast_filt_eff / cast_norm
        return {**data, 'weight': weight}
    return _calculate_weights


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.save_path, exist_ok=True)
    files = []
    sizes = []
    weight_info = []

    jz_description = pd.read_csv(args.jz_description_file)
    print(jz_description)
    lumi = 139_000

    for jz in range(args.min_jz, args.max_jz + 1):
        file = os.path.join(args.load_path, f'JZ{jz}', args.dataset_type)
        filt_eff = jz_description[jz_description['JZ'] == jz]['filtEff'].values[0]
        cross_section = jz_description[jz_description['JZ'] == jz]['crossSection [pb]'].values[0]
        norm_name = jz_description[jz_description['JZ'] == jz]['Norm'].values[0]

        dataset = JIDENNDataset.load(file)
        size = dataset.length
        norm = dataset.metadata[norm_name]

        cross_section = tf.constant(cross_section)
        filt_eff = tf.constant(filt_eff)
        lumi = tf.constant(lumi)
        norm = tf.constant(norm)
        # jz = tf.constant(jz, dtype=tf.int32)
        weight_info.append((cross_section, filt_eff, lumi, norm, jz))
        sizes.append(size)
        files.append(file)

    sampling_weights = [size / sum(sizes) for size in sizes]
    print(sampling_weights)

    @tf.function
    def jz_cutter(dataset: tf.data.Dataset, weight_info) -> tf.data.Dataset:
        cross_section, filt_eff, lumi, norm, jz = weight_info
        dataset = dataset.filter(get_cut_fn('jets_pt', upper_limit=MAX_FRAC * tf.constant(JZ_HIGH_PT)[jz - 1]))
        dataset = dataset.map(write_weights(cross_section, filt_eff, lumi, norm))
        dataset = dataset.map(write_new_variable(variable_name='JZ_slice',
                              variable_value=tf.constant(jz, dtype=tf.int32)))
        if jz == 4:
            dataset = dataset.skip(10_000_000)
        return dataset

    dataset = JIDENNDataset.load_multiple(files, dataset_mapper=jz_cutter,
                                          file_labels=weight_info, weights=sampling_weights)
    dataset = dataset.take(args.take)
    dataset = dataset.apply(partial(flatten_dataset, reference_variable=args.reference_variable,
                                    wanted_values=args.wanted_values, variable='jets_eta', lower_cut=-args.eta_cut, upper_cut=args.eta_cut))
    # dataset = dataset.filter(lambda x: tf.random.uniform([]) < TAKE_FRAC)

    dataset = dataset.apply(lambda x: x.shuffle(args.shuffle).prefetch(tf.data.AUTOTUNE))
    dataset.save(args.save_path, num_shards=args.num_shards)
    logging.info(f'Saved dataset to {args.save_path}')
    dataset = JIDENNDataset.load(args.save_path)
    size = dataset.length
    print(f"Number of jets: {size}")
    dataset.plot_single_variable('jets_pt',
                                 weight_variable='weight',
                                 save_path=os.path.join(args.save_path, 'pt.png'),
                                 ylog=True,
                                 badge_text='$N_{\mathrm{jets}}$ = ' + f'{size:,} \n',
                                 bins=100,
                                 multiple='stack',
                                 hue_variable='JZ_slice')
    dataset.plot_single_variable('jets_pt',
                                 weight_variable=None,
                                 save_path=os.path.join(args.save_path, 'pt_noW.png'),
                                 ylog=True,
                                 badge_text='$N_{\mathrm{jets}}$ = ' + f'{size:,} \n',
                                 bins=100,
                                 multiple='stack',
                                 hue_variable='JZ_slice')
    dataset.plot_single_variable('jets_pt',
                                 weight_variable='weight',
                                 save_path=os.path.join(args.save_path, 'pt_noJZ.png'),
                                 ylog=True,
                                 badge_text='$N_{\mathrm{jets}}$ = ' + f'{size:,} \n',
                                 bins=100,)
    dataset.plot_single_variable('jets_pt',
                                 weight_variable='weight',
                                 save_path=os.path.join(args.save_path, 'pt_label.png'),
                                 ylog=True,
                                 badge_text='$N_{\mathrm{jets}}$ = ' + f'{size:,} \n',
                                 bins=100,
                                 multiple='stack',
                                 hue_variable='jets_PartonTruthLabelID')

    dataset.plot_single_variable('weight',
                                 save_path=os.path.join(args.save_path, 'weight.png'),
                                 ylog=True,
                                 bins=100,
                                 multiple='stack',
                                 hue_variable='JZ_slice')


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
