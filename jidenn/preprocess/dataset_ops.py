import tensorflow as tf
import os
import argparse
import pickle
from typing import Tuple, List, Dict, Union, Optional

ROOTVariables = Dict[str, tf.RaggedTensor]


def load_dataset(file_path: str, element_spec_path: Optional[str] = None) -> tf.data.Dataset:
    """Load a dataset from a file path.

    Args:
        file_path (str): Path to the saved dataset with tf.data.Dataset.save() 
        element_spec_path (str, optional): Path to the element spec pickle file. Defaults to None.
            If None, the element spec will be loaded from file_path/element_spec.

    Returns:
        tf.data.Dataset: The loaded dataset
    """
    element_spec = None
    if element_spec_path is not None:
        with open(element_spec_path, 'rb') as f:
            element_spec = pickle.load(f)
    else:
        try:
            with open(os.path.join(file_path, 'element_spec'), 'rb') as f:
                element_spec = pickle.load(f)
        except FileNotFoundError:
            print("No element spec found, graph mode will not work")

    dataset = tf.data.Dataset.load(file_path, compression='GZIP', element_spec=element_spec)
    return dataset


def split_train_dev_test(dataset: tf.data.Dataset,
                         train_fraction: float,
                         dev_fraction: float,
                         test_fraction: float) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Split a dataset into train, dev and test sets. The fractions must sum to 1.0 and only 2 decimal places 
    are taken into account. The cardinality of the returned datasets might not be known as the spliting is done
    by random filtering.

    Args:
        dataset (tf.data.Dataset): The dataset to split
        train_fraction (float): The fraction of the dataset to use for training dataset.
        dev_fraction (float): The fraction of the dataset to use for development dataset.
        test_fraction (float): The fraction of the dataset to use for testing dataset.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: The train, dev and test datasets.
    """

    train_fraction = round(train_fraction, 2)
    dev_fraction = round(dev_fraction, 2)
    test_fraction = round(test_fraction, 2)

    assert train_fraction + dev_fraction + test_fraction == 1.0, "Fractions must sum to 1.0 and can only have 2 decimal places."

    @tf.function
    def random_number() -> tf.Tensor:
        return tf.random.uniform(shape=[], minval=1, maxval=100, dtype=tf.int32)

    @tf.function
    def train_filter(sample: ROOTVariables, random_number: tf.Tensor) -> tf.Tensor:
        return tf.less_equal(random_number, train_fraction * 100)

    @tf.function
    def dev_filter(sample: ROOTVariables, random_number: tf.Tensor) -> tf.Tensor:
        return tf.greater(random_number, train_fraction * 100) and tf.less_equal(random_number, (train_fraction + dev_fraction) * 100)

    @tf.function
    def test_filter(sample: ROOTVariables, random_number: tf.Tensor) -> tf.Tensor:
        return tf.greater(random_number, (train_fraction + dev_fraction) * 100)

    @tf.function
    def delete_random_number(sample: ROOTVariables, random_number: tf.Tensor) -> ROOTVariables:
        return sample

    return (
        dataset.map(random_number).filter(train_filter).map(delete_random_number),
        dataset.map(random_number).filter(dev_filter).map(delete_random_number),
        dataset.map(random_number).filter(test_filter).map(delete_random_number)
    )


def save_dataset(dataset: tf.data.Dataset, file_path: str, num_shards: int = 256) -> None:
    """Save a dataset to a file path, with uniform sharding. The dataset will 
    be saved in the GZIP format. The element spec will be saved in a pickle file
    in the same directory as the dataset with name `element_spec`.

    Args:
        dataset (tf.data.Dataset): The dataset to save
        file_path (str): The path to save the dataset to
        num_shards (int, optional): The number of shards to use. Defaults to 256.

    """

    @tf.function
    def gen_random_number(sample: ROOTVariables) -> tf.Tensor:
        return tf.random.uniform(shape=[], minval=0, maxval=num_shards, dtype=tf.int64)

    dataset.save(file_path, compression='GZIP', shard_func=gen_random_number)

    with open(os.path.join(file_path, 'element_spec'), 'wb') as f:
        pickle.dump(dataset.element_spec, f)
