from __future__ import annotations
import tensorflow as tf
from typing import Callable
import pickle
import os
import uproot
import logging
import pandas as pd
import awkward as ak
#

logging.basicConfig(format='[%(asctime)s][%(levelname)s] - %(message)s',
                    level=logging.INFO,  datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__name__)

ROOTVariables = dict[str, tf.RaggedTensor]


class ROOTDataset:
    def __init__(self, dataset: tf.data.Dataset, variables: list[str]):
        # for dt in dataset.take(1):
        #     if not isinstance(dt, dict):
        #         raise TypeError("ROOTDataset only accepts datasets that yield dictionaries")
        self._variables = variables
        self._dataset = dataset

    @property
    def variables(self) -> list[str]:
        return self._variables

    @property
    def dataset(self) -> tf.data.Dataset:
        return self._dataset

    @classmethod
    def from_root_file(cls, filename: str,
                       tree_name: str = 'NOMINAL',
                       metadata_hist: str = 'h_metadata') -> ROOTDataset:

        log.info(f"Loading ROOT file {filename}")
        file = uproot.open(filename, object_cache=None, array_cache=None)
        log.info(f"Getting tree {tree_name}")
        tree = file[tree_name]
        variables = tree.keys()
        # tree_iterator = tree.iterate(entrysteps=100000, flatten=False)

        sample = {}
        for var in variables:
            # log.info(f"Getting variable {var}")
            array: ak.Array = tree[var].array(library="ak")
            if ak.size(ak.flatten(array, axis=None)) == 0:
                continue
            sample[var] = cls.awkward_to_tensor(array)

        if metadata_hist is not None:
            log.info("Getting metadata")
            metadata = file[metadata_hist].values()
            sample['metadata'] = tf.tile(tf.constant(metadata)[tf.newaxis, :], [sample['eventNumber'].shape[0], 1])

        log.info(f'Done loading file:{filename}')
        dataset = tf.data.Dataset.from_tensors(sample)
        return cls(dataset, variables)

    @classmethod
    def awkward_to_tensor(cls, array: ak.Array) -> tf.RaggedTensor:
        if array.ndim == 1:
            return tf.constant(array.to_list())
        elif array.ndim == 2:
            row_lengths = ak.num(array, axis=1).to_list()
            return tf.RaggedTensor.from_row_lengths(ak.flatten(array, axis=None).to_list(), row_lengths=row_lengths, validate=False)
        else:
            nested_row_lengths = [ak.flatten(ak.num(array, axis=ax), axis=None).to_list()
                                  for ax in range(1, array.ndim)]
            return tf.RaggedTensor.from_nested_row_lengths(ak.flatten(
                array, axis=None).to_list(), nested_row_lengths=nested_row_lengths, validate=False)

    @classmethod
    def _concat(cls, datasets: list[ROOTDataset]) -> ROOTDataset:
        for dataset in datasets:
            if dataset.variables != datasets[0].variables:
                raise ValueError("Variables of datasets do not match")
        final_dataset = datasets[0]._dataset
        for ds in datasets[1:]:
            final_dataset = final_dataset.concatenate(ds._dataset)
        return cls(final_dataset, datasets[0].variables)

    @classmethod
    def from_root_files(cls, filenames: list[str] | str) -> ROOTDataset:
        if isinstance(filenames, str):
            filenames = [filenames]
        return cls._concat([cls.from_root_file(filename) for filename in filenames])

    @classmethod
    def load(cls, file: str, element_spec_path: str = None) -> ROOTDataset:
        element_spec_path = os.path.join(file, 'element_spec') if element_spec_path is None else element_spec_path
        with open(element_spec_path, 'rb') as f:
            element_spec = pickle.load(f)
        dataset = tf.data.experimental.load(file, compression='GZIP', element_spec=element_spec)
        return cls(dataset, list(element_spec.keys()))

    def save(self, save_path: str, element_spec_path: str = None, shard_func: Callable[[ROOTVariables], tf.Tensor] = None) -> None:
        element_spec_path = os.path.join(save_path, 'element_spec') if element_spec_path is None else element_spec_path
        element_spec = self._dataset.element_spec
        tf.data.experimental.save(self._dataset, save_path, compression='GZIP', shard_func=shard_func)
        with open(element_spec_path, 'wb') as f:
            pickle.dump(element_spec, f)

    def split_by_size(self, size: float) -> tuple[ROOTDataset, ROOTDataset]:
        return ROOTDataset(self._dataset.take(int(size * self._dataset.cardinality().numpy())), self._variables), ROOTDataset(self._dataset.skip(int(size * self._dataset.cardinality().numpy())), self._variables)

    def split_train_dev_test(self, test_size: float, dev_size: float) -> tuple[ROOTDataset, ROOTDataset, ROOTDataset]:
        train_size = 1 - test_size - dev_size
        train, dev_test = self.split_by_size(train_size)
        dev, test = dev_test.split_by_size(dev_size / (1 - train_size))
        return train, dev, test
