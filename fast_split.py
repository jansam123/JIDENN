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
parser.add_argument("--num_shards", type=int, default=96, help="Path to the root file")

tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)

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
    path = '/scratch/ucjf-atlas/jankovys/JIDENN/dataset3/H7EG_jetjet_JZ1/all'
    save_path = '/scratch/ucjf-atlas/jankovys/JIDENN/dataset3/H7EG_jetjet_JZ1/'
    
    with open(os.path.join(path, 'element_spec'), 'rb') as f:
        element_spec = pickle.load(f)
    dataset = tf.data.Dataset.load(path, compression='GZIP', element_spec=element_spec)
    dataset = dataset.map(gen_random_number)
    
    @tf.function
    def random_shards(sample: ROOTVariables) -> tf.Tensor:
        return tf.random.uniform(shape=[], minval=0, maxval=args.num_shards, dtype=tf.int64)
    
    train_dataset = dataset.filter(filter_train).map(delete_random_number).prefetch(tf.data.AUTOTUNE)
    test_dataset = dataset.filter(filter_test).map(delete_random_number).prefetch(tf.data.AUTOTUNE)
    dev_dataset = dataset.filter(filter_dev).map(delete_random_number).prefetch(tf.data.AUTOTUNE)
    
    for name, ds in zip(['train', 'test', 'dev'], [train_dataset, test_dataset, dev_dataset]):
        logging.info(f"Saving {name} dataset to {save_path}/{name}")
        
        save_path = os.path.join(save_path, name)
        os.makedirs(save_path, exist_ok=True)
        
        with open(os.path.join(save_path, 'element_spec'), 'wb') as f:
            pickle.dump(ds.element_spec, f)
        ds.save(path=save_path, compression='GZIP', shard_func=random_shards)
        
        ds = tf.data.Dataset.load(save_path, compression='GZIP')
        for i in ds.take(1):
            print(i)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
