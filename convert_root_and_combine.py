from jidenn.preprocess.resampling import write_new_variable
from jidenn.preprocess.flatten_dataset import flatten_dataset
from jidenn.data.JIDENNDataset import JIDENNDataset
from typing import Callable
import os
import argparse
import logging
import tensorflow as tf
import pickle
import hashlib
import pandas as pd
import ray
import subprocess
logging.basicConfig(format='[%(asctime)s][%(levelname)s] - %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
#


parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, help="Path to save the dataset")
parser.add_argument("--load_path", type=str, help="Path to the root file")
parser.add_argument("--backend", type=str, default='pd',
                    help="Backend to use for loading the dataset")
parser.add_argument("--max_size", type=int, default=None,
                    help="Maximum number of samples to use")
parser.add_argument("--train_frac", type=float, default=0.8,
                    help="Fraction of the dataset to use for training")
parser.add_argument("--dev_frac", type=float, default=0.1,
                    help="Fraction of the dataset to use for development")
parser.add_argument("--test_frac", type=float, default=0.1,
                    help="Fraction of the dataset to use for testing")
# parser.add_argument("--reference_variable", type=str, default='jets_PartonTruthLabelID',
#                     help="Variable to use as reference for flattening")
# parser.add_argument("--wanted_values", type=int, nargs='+',
#                     default=[1, 2, 3, 4, 5, 6, 21], help="Values to keep in the reference variable")
parser.add_argument("--shuflle", type=int, default=100,
                    required=False, help="Shuffle buffer size")
parser.add_argument("--num_shards", type=int, default=256, required=False,
                    help="Number of shards to save the dataset in")
parser.add_argument("--take", type=int, default=None,
                    required=False, help="Number of samples to take")
parser.add_argument("--jz_description_file", type=str,
                    default='data/sample_description.csv', help="JZ description csv file.")
parser.add_argument("--from_cache", action='store_true',
                    help="Load from cache")
parser.add_argument("--mode", type=str, default='parallel',
                    help="Mode to use when loading multiple files")


def write_weights(cross_section: float = 1., filt_eff: float = 1., lumi: float = 1., norm: float = 1.) -> Callable:
    @tf.function
    def _calculate_weights(data):
        w = tf.cast(data['weight_mc'], tf.float64)
        cast_lumi = tf.cast(lumi, tf.float64)
        cast_cross_section = tf.cast(cross_section, tf.float64)
        cast_filt_eff = tf.cast(filt_eff, tf.float64)
        cast_norm = tf.cast(norm, tf.float64)
        weight = w[0] * cast_lumi * \
            cast_cross_section * cast_filt_eff / cast_norm
        return {**data, 'weight': weight}
    return _calculate_weights


def main(args: argparse.Namespace) -> None:
    files = [os.path.join(args.load_path, file) for file in os.listdir(
        args.load_path) if file.endswith('.root')]
    old_tmp_dir = os.path.join(
        args.save_path.replace('all_MCs', 'all'), 'tmp')
    dssid = args.save_path.split('/')[7]
    args.save_path = '/'.join(args.save_path.split('/')[:-1])

    jz_description = pd.read_csv(args.jz_description_file, index_col=0)
    jz_description.index = jz_description.index.astype(str)
    jz = jz_description.loc[dssid]['JZ']
    filt_eff = jz_description.loc[dssid]['filtEff']
    cross_section = jz_description.loc[dssid]['crossSection [pb]']
    norm_name = jz_description.loc[dssid]['Norm']
    name = jz_description.loc[dssid]['Description']
    name = name[:name.rfind('_')]
    lumi = 139_000

    logging.info('JZ description pd.DataFrame:')
    logging.info(jz_description)
    logging.info(f'name: {name}')
    logging.info(f'JZ: {jz}')
    logging.info(f'filt_eff: {filt_eff}')
    logging.info(f'cross_section: {cross_section}')
    logging.info(f'Processing dataset dssid: {dssid}')

    args.save_path = os.path.join(args.save_path, name, f'JZ{jz}')
    tmp_dir = os.path.join(args.save_path, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)

    logging.info(f'Old tmp dir: {old_tmp_dir}')
    logging.info(f'Tmp dir: {tmp_dir}')
    logging.info(f'Saving dataset to: {args.save_path}')

    total_size = 0

    ds_file_names = []

    # @ray.remote
    # class DataSplitter:
    #     def __init__(self, file):
    #         self.file = file
    #         self.tmp_file_name = os.path.join(
    #             tmp_dir, os.path.basename(file).replace('.root', ''))

    #     def _create_dataset(self):
    #         return JIDENNDataset.from_root_file(self.file, backend=args.backend)

    #     def save(self):
    #         try:
    #             with open(os.path.join(tmp_dir, hashlib.sha256(str(self.tmp_file_name).encode('utf-8')).hexdigest() + '.pkl'), 'rb') as f:
    #                 name = pickle.load(f)
    #                 if name == self.tmp_file_name:
    #                     logging.info(f'Found cached file: {self.file}')
    #                     return name
    #         except FileNotFoundError:
    #             try:
    #                 dataset = self._create_dataset()
    #                 if len(dataset.metadata.keys()) == 0:
    #                     logging.error(
    #                         f'Failed to load file: {self.file}, empty metadata: {dataset.metadata}')
    #             except Exception as e:
    #                 logging.error(
    #                     f'Failed to load file: {self.file} with error: {e}')
    #                 return None

    #             dataset.save(self.tmp_file_name)
    #             with open(os.path.join(tmp_dir, hashlib.sha256(str(self.tmp_file_name).encode('utf-8')).hexdigest() + '.pkl'), 'wb') as f:
    #                 pickle.dump(self.tmp_file_name, f)
    #             return self.tmp_file_name

    # with ray.init(num_cpus=1):  # limit your number of parallel processes
    #     actors = [DataSplitter.remote(file=file) for file in files]
    #     paths = []
    #     for id, actor in enumerate(actors):
    #         try:
    #             paths.append(ray.get(actor.save.remote()))
    #         except ray.exceptions.RayActorError as ex:
    #             logging.error(f'Out of memory error: {ex}')
    #             logging.error(f'Failed to load file: {files[id]}')
    #             continue
    if not args.from_cache:
        for file in files:
            file_size = os.path.getsize(file)*1e-9
            logging.info(f'File size: {file_size:.2f} GB')
            logging.info(f'Processing file: {file}')
            tmp_file_name = os.path.join(
                tmp_dir, os.path.basename(file).replace('.root', ''))
            try:
                with open(os.path.join(tmp_dir, hashlib.sha256(str(tmp_file_name).encode('utf-8')).hexdigest() + '.pkl'), 'rb') as f:
                    name = pickle.load(f)
                    if name == tmp_file_name:
                        ds_file_names.append(name)
                        logging.info(f'Found cached file: {file}')
                        continue
            except FileNotFoundError:
                pass

            if file_size > 9:
                logging.warning(f'File size too large, skipping...')
                continue
            logging.info(f'Found file: {file}')
            try:
                ds = JIDENNDataset.from_root_file(file, backend=args.backend)
                if len(ds.metadata.keys()) == 0:
                    logging.error(
                        f'Failed to load file: {file}, empty metadata: {ds.metadata}')
                    continue
            except Exception as e:
                logging.error(f'Failed to load file: {file} with error: {e}')
                continue
            total_size += ds.length
            logging.info(f'Current size: {total_size}')
            if args.max_size is not None and total_size > args.max_size:
                break
            ds.save(tmp_file_name)
            ds_file_names.append(tmp_file_name)

            with open(os.path.join(tmp_dir, hashlib.sha256(str(tmp_file_name).encode('utf-8')).hexdigest() + '.pkl'), 'wb') as f:
                pickle.dump(tmp_file_name, f)
            del ds
        raise Exception('Stop here')

    else:
        logging.info('Loading from cache...')
        ds_file_names = []
        total_size = 0
        for file in files:
            tmp_file_name = os.path.join(
                tmp_dir, os.path.basename(file).replace('.root', ''))
            if os.path.exists(os.path.join(tmp_dir, tmp_file_name)) and 'element_spec.pkl' in os.listdir(os.path.join(tmp_dir, tmp_file_name)):
                if args.max_size is not None:
                    file_size = subprocess.check_output(
                        ['du', '-s', tmp_file_name]).split()[0].decode('utf-8')
                    file_size = float(file_size)/1024**2
                    total_size += file_size
                    if total_size > args.max_size:
                        break
                ds_file_names.append(tmp_file_name)
        logging.info(f'Found files: {len(ds_file_names)}')

    if args.mode == 'parallel':
        dataset = JIDENNDataset.load_parallel(ds_file_names, file_labels=None)
    elif args.mode == 'concatenate':
        dataset = JIDENNDataset.load_multiple(
            ds_file_names, file_labels=None, mode='concatenate')
    elif args.mode == 'interleave':
        dataset = JIDENNDataset.load_multiple(
            ds_file_names, file_labels=None, mode='interleave')

    norm = dataset.metadata[norm_name]

    # dataset = dataset.apply(partial(flatten_dataset, reference_variable=args.reference_variable,
    #                         wanted_values=args.wanted_values))
    dataset = dataset.remap_data(
        write_new_variable(jz, variable_name='JZ_slice'))
    dataset = dataset.remap_data(write_weights(
        cross_section, filt_eff, lumi, norm))
    dataset = dataset.apply(lambda x: x.prefetch(tf.data.AUTOTUNE))

    dss = dataset.split_train_dev_test(
        args.train_frac, args.dev_frac, args.test_frac, backend='coin')

    for name, ds in zip(['train', 'dev', 'test'], dss):
        if os.path.exists(os.path.join(args.save_path, name)):
            os.system(f'rm -rf {os.path.join(args.save_path, name)}')
        ds.save(os.path.join(args.save_path, name), num_shards=args.num_shards)
        logging.info(f'Saved dataset: {name}')

    # dataset.take(1_000_000).plot_single_variable(variable='jets_pt',
    #                                              save_path=os.path.join(args.save_path, 'pt.png'),
    #                                              weight_variable='weight',
    #                                              bins=200)

    # os.system(f'rm -rf {tmp_dir}')


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
