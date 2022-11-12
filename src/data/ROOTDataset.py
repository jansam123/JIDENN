from __future__ import annotations
import tensorflow as tf
from typing import Callable
import pickle
import os
import uproot
#
from .utils.CutV2 import Cut
# import src.config.config_subclasses as cfg
# import pandas as pd
# import time
# import os


ROOTVariables = dict[str, tf.RaggedTensor]


class ROOTDataset:

    def __init__(self, dataset: tf.data.Dataset, variables: list[str]):
        for dt in dataset.take(1):
            if not isinstance(dt, dict):
                raise TypeError("ROOTDataset only accepts datasets that yield dictionaries")
        self._variables = variables
        self._dataset = dataset

    @property
    def variables(self) -> list[str]:
        return self._variables

    @property
    def dataset(self) -> tf.data.Dataset:
        return self._dataset

    @classmethod
    def from_root_file(cls, filename: str, transformation: Callable[[ROOTVariables], ROOTVariables] = None, tree_name: str = 'NOMINAL', metadata_hist: str = 'h_metadata') -> ROOTDataset:
        file = uproot.open(filename)
        tree = file[tree_name]
        variables = tree.keys()
        sample = {}
        for var in variables:
            df = tree[var].array(library="pd")
            if df.empty:
                continue
            elif df.index.nlevels == 1 and df.dtypes == 'object':
                df = df.explode()
                value_rowids = df.index.get_level_values(0).to_numpy()
                df = df.reset_index(drop=True).explode()
                value_rowids_2 = df.index.get_level_values(0).to_numpy()
                sample[var] = tf.RaggedTensor.from_nested_value_rowids(df.values.tolist(), nested_value_rowids=[
                                                                       value_rowids, value_rowids_2], validate=False)
            elif df.index.nlevels == 1 and df.dtypes != 'object':
                sample[var] = tf.constant(df)
            elif df.index.nlevels > 1:
                value_rowids = df.index.get_level_values(0).values
                sample[var] = tf.RaggedTensor.from_value_rowids(df.values, value_rowids, validate=False)

        sample = transformation(sample) if transformation is not None else sample
        if metadata_hist is not None:
            metadata = file[metadata_hist].values()
            sample['metadata'] = tf.tile(tf.constant(metadata)[tf.newaxis, :], [sample['eventNumber'].shape[0], 1])
        return cls(tf.data.Dataset.from_tensor_slices(sample), variables)

    @classmethod
    def concat(cls, datasets: list[ROOTDataset]) -> ROOTDataset:
        for dataset in datasets:
            if dataset.variables != datasets[0].variables:
                raise ValueError("Variables of datasets do not match")
        final_dataset = datasets[0]._dataset
        for ds in datasets[1:]:
            final_dataset = final_dataset.concatenate(ds._dataset)
        return cls(final_dataset, datasets[0].variables)

    @classmethod
    def from_root_files(cls, filenames: list[str] | str, transformation: Callable[[ROOTVariables], ROOTVariables] = None) -> ROOTDataset:
        if isinstance(filenames, str):
            filenames = [filenames]
        return cls.concat([cls.from_root_file(filename, transformation) for filename in filenames])

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

    def filter(self, cut: Cut) -> ROOTDataset:
        return ROOTDataset(self._dataset.filter(cut), self._variables)

    def split_by_size(self, size: float) -> tuple[ROOTDataset, ROOTDataset]:
        return ROOTDataset(self._dataset.take(int(size * self._dataset.cardinality().numpy())), self._variables), ROOTDataset(self._dataset.skip(int(size * self._dataset.cardinality().numpy())), self._variables)

    def split_train_dev_test(self, test_size: float, dev_size: float) -> tuple[ROOTDataset, ROOTDataset, ROOTDataset]:
        train_size = 1 - test_size - dev_size
        train, dev_test = self.split_by_size(train_size)
        dev, test = dev_test.split_by_size(dev_size / (1 - train_size))
        return train, dev, test
