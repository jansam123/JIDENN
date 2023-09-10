from typing import Callable
import os
import argparse
import logging
from functools import partial
import tensorflow as tf
import pandas as pd
logging.basicConfig(format='[%(asctime)s][%(levelname)s] - %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
#

from jidenn.data.JIDENNDataset import JIDENNDataset
from jidenn.preprocess.flatten_dataset import flatten_dataset
from jidenn.preprocess.resampling import write_new_variable


parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, help="Path to save the dataset")
parser.add_argument("--load_dir", type=str, help="Path to the root file")
parser.add_argument("--backend", type=str, default='pd', help="Backend to use for loading the dataset")
parser.add_argument("--max_size", type=int, default=10_000, help="Maximum number of samples to use")
parser.add_argument("--reference_variable", type=str, default='jets_PartonTruthLabelID',
                    help="Variable to use as reference for flattening")
parser.add_argument("--wanted_values", type=int, nargs='+',
                    default=[1, 2, 3, 4, 5, 6, 21], help="Values to keep in the reference variable")
parser.add_argument("--shuflle", type=int, default=100_000, required=False, help="Shuffle buffer size")
parser.add_argument("--num_shards", type=int, default=4, required=False,
                    help="Number of shards to save the dataset in")
parser.add_argument("--jz_description_file", type=str,
                    default='data/JZ_description_all.csv', help="JZ description csv file.")


def write_weights(cross_section: float = 1., filt_eff: float = 1., lumi: float = 1., norm: float = 1.) -> Callable:
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
    files = [os.path.join(args.load_dir, file) for file in os.listdir(args.load_dir) if file.endswith('.root')]
    # shuffle the files with python native shuffle:
    files = files[::-1]
    tmp_dir = os.path.join(args.save_path, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)

    dssid = args.save_path.split('/')[6].split('.')[0]
    jz_description = pd.read_csv(args.jz_description_file, index_col=0)
    # change the index type to str from int:
    jz_description.index = jz_description.index.astype(str)
    filt_eff = jz_description.loc[dssid]['filtEff']
    cross_section = jz_description.loc[dssid]['crossSection [pb]']
    lumi = 139_000

    total_size = 0
    ds_file_names = []
    for file in files:
        logging.info(f'Found file: {file}')
        try:
            ds = JIDENNDataset.from_root_file(file, backend=args.backend)
            try:
                ds.metadata[0]
            except IndexError:
                logging.error(f'Failed to load file: {file} with error: {ds.metadata}')
                continue
        except Exception as e:
            logging.error(f'Failed to load file: {file} with error: {e}')
            continue
        total_size += ds.length
        logging.info(f'Current size: {total_size}')
        if total_size > args.max_size:
            break
        tmp_file_name = os.path.join(tmp_dir, os.path.basename(file).replace('.root', ''))
        ds.save(tmp_file_name)
        ds_file_names.append(tmp_file_name)

    dataset = JIDENNDataset.load_parallel(ds_file_names, file_labels=None)
    norm = dataset.metadata['sum_AOD_w']

    # dataset = dataset.apply(partial(flatten_dataset, reference_variable=args.reference_variable,
    #                         wanted_values=args.wanted_values))
    dataset = dataset.remap_data(write_new_variable(dssid, variable_name='dssid'))
    dataset = dataset.remap_data(write_weights(cross_section, filt_eff, lumi, norm))
    dataset = dataset.apply(lambda x: x.shuffle(args.shuflle).prefetch(tf.data.AUTOTUNE))
    dataset.save(os.path.join(args.save_path, 'test'), num_shards=args.num_shards)
    dataset.plot_single_variable(variable='jets_pt',
                                 save_path=os.path.join(args.save_path, 'pt.png'),
                                 weight_variable='weight',
                                 bins=200)
    os.system(f'rm -rf {tmp_dir}')


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
