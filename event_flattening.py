#!/home/jankovys/JIDENN/venv/bin/python
import tensorflow as tf
import os
import argparse
import pickle
import logging
from typing import Tuple, List, Dict, Union, Optional
#
logging.basicConfig(level=logging.INFO)
# from src.data.ROOTDataset import ROOTDataset, ROOTVariables
ROOTVariables = dict[str, tf.RaggedTensor]

parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, help="Path to save the dataset")
parser.add_argument("--file_path", type=str, help="Path to the root file")
parser.add_argument("--num_shards", type=int, default=96, help="Path to the root file")

tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)


@tf.function
def no_pile_up(sample: ROOTVariables) -> ROOTVariables:
    sample = sample.copy()
    jet_mask = tf.math.greater(sample['jets_PartonTruthLabelID'], 0)
    for key, item in sample.items():
        if key.startswith('jets_') and key != 'jets_n':
            sample[key] = tf.ragged.boolean_mask(item, jet_mask)
        sample[key] = tf.cast(sample[key], tf.int32 if 'int' in item.dtype.name else tf.float32)
    return sample


@tf.function
def filter_empty_jets(sample: ROOTVariables) -> tf.Tensor:
    return tf.greater(tf.size(sample['jets_PartonTruthLabelID']), 0)


@tf.function
def flatten_toJet(sample: ROOTVariables) -> ROOTVariables:
    sample = sample.copy()
    jet_shape = tf.shape(sample['jets_PartonTruthLabelID'])
    for key, item in sample.items():
        if isinstance(item, tf.RaggedTensor):
            continue
        elif len(tf.shape(item)) == 0:
            sample[key] = tf.tile(item[tf.newaxis, tf.newaxis], [jet_shape[0], 1])
        elif tf.shape(item)[0] != jet_shape[0]:
            sample[key] = tf.tile(item[tf.newaxis, :], [jet_shape[0], 1])
        else:
            continue
    return tf.data.Dataset.from_tensor_slices(sample)

@tf.function
def gen_random_number(sample: ROOTVariables) -> Tuple[ROOTVariables, tf.Tensor]:
    return sample, tf.random.uniform(shape=[], minval=0, maxval=9, dtype=tf.int32)

@tf.function
def filter_test(sample: ROOTVariables, random_number: tf.Tensor) -> tf.Tensor:
    return tf.equal(random_number, 0)

@tf.function
def filter_dev(sample: ROOTVariables, random_number: tf.Tensor) -> tf.Tensor:
    return tf.equal(random_number, 1)

@tf.function
def filter_train(sample: ROOTVariables, random_number: tf.Tensor) -> tf.Tensor:
    return tf.not_equal(random_number, 0) and tf.not_equal(random_number, 1)

@tf.function
def delete_random_number(sample: ROOTVariables, random_number: tf.Tensor) -> ROOTVariables:
    return sample

def main(args: argparse.Namespace) -> None:
    os.makedirs(args.save_path, exist_ok=True)
    base_dir = args.file_path
    base_files = [f for f in os.listdir(base_dir) if f.startswith('_')]
    files = [os.path.join(base_dir, f) for f in base_files]
    logging.info(f"Found {len(files)} files")
    logging.info(f"Files: {files}")

    def load_dataset_file(element_spec, file: str) -> tf.data.Dataset:
        root_dt = tf.data.Dataset.load(file, compression='GZIP', element_spec=element_spec)
        root_dt = root_dt.map(no_pile_up).filter(filter_empty_jets)
        return root_dt
    
    @tf.function
    def random_shards(x: ROOTVariables) -> tf.Tensor:
        return tf.random.uniform(shape=[], minval=0, maxval=args.num_shards, dtype=tf.int64)
    
    try:
        with open(os.path.join(files[0], 'element_spec'), 'rb') as f:
            element_spec = pickle.load(f)
    except FileNotFoundError:
        files[0] = os.path.join(files[0], base_files[0])
        with open(os.path.join(files[0], 'element_spec'), 'rb') as f:
            element_spec = pickle.load(f)
            
    dataset = load_dataset_file(element_spec, files[0])
    
    for file, b_file in zip(files[1:], base_files[1:]):
        try:
            with open(os.path.join(file, 'element_spec'), 'rb') as f:
                element_spec = pickle.load(f)
        except FileNotFoundError:
            file = os.path.join(file, b_file)
            with open(os.path.join(file, 'element_spec'), 'rb') as f:
                element_spec = pickle.load(f)
        dataset = dataset.concatenate(load_dataset_file(element_spec, file))
        
    dataset = dataset.interleave(flatten_toJet)
    dataset = dataset.shuffle(1000, reshuffle_each_iteration=False)
    dataset = dataset.map(gen_random_number)
    
    train_dataset = dataset.filter(filter_train).map(delete_random_number).prefetch(tf.data.AUTOTUNE)
    test_dataset = dataset.filter(filter_test).map(delete_random_number).prefetch(tf.data.AUTOTUNE)
    dev_dataset = dataset.filter(filter_dev).map(delete_random_number).prefetch(tf.data.AUTOTUNE)
    
    for name, ds in zip(['train', 'test', 'dev'], [train_dataset, test_dataset, dev_dataset]):
        logging.info(f"Saving {name} dataset to {args.save_path}/{name}")
        
        save_path = os.path.join(args.save_path, name)
        os.makedirs(save_path, exist_ok=True)
        
        with open(os.path.join(save_path, 'element_spec'), 'wb') as f:
            pickle.dump(ds.element_spec, f)
        ds.save(path=save_path, compression='GZIP', shard_func=random_shards)
        
        ds = tf.data.Dataset.load(save_path, compression='GZIP')
        print(ds.cardinality().numpy())
        for i in ds.take(1):
            print(i)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
