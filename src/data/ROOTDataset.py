from __future__ import annotations
import tensorflow as tf
from typing import Callable, List, Union, Dict
import pickle
import os
import uproot
import logging
import pandas as pd
import awkward as ak
#
from .utils.conversions import pandas_to_tensor, awkward_to_tensor


ROOTVariables = Dict[str, tf.RaggedTensor]


class ROOTDataset:
    """
    The ROOTDataset class represents a dataset of ROOT files.

    It contains methods to load the files and access their variables as tensorflow datasets.

    Methods:
        from_root_file(filename: str, tree_name: str, metadata_hist: str, backend: str):
        Loads a single ROOT file and returns a ROOTDataset object.
        from_root_files(filenames: Union[List[str], str]):
        Loads a list of ROOT files and returns a concatenated ROOTDataset object.
        load(file: str, element_spec_path: str):
        Loads a saved ROOTDataset object from disk.
        save(save_path: str, element_spec_path: str, shard_func: Callable[[ROOTVariables], tf.Tensor]):
        Saves the ROOTDataset object to disk as a tensorflow dataset.
    """

    def __init__(self, dataset: tf.data.Dataset, variables: List[str]):
        # for dt in dataset.take(1):
        #     if not isinstance(dt, dict):
        #         raise TypeError("ROOTDataset only accepts datasets that yield dictionaries")
        self._variables = variables
        self._dataset = dataset

    @property
    def variables(self) -> List[str]:
        return self._variables

    @property
    def dataset(self) -> tf.data.Dataset:
        return self._dataset

    @classmethod
    def from_root_file(cls, filename: str,
                       tree_name: str = 'NOMINAL',
                       metadata_hist: str = 'h_metadata',
                       backend: str = 'pd') -> ROOTDataset:
        file = uproot.open(filename, object_cache=None, array_cache=None)
        tree = file[tree_name]

        logging.info(f"Loading ROOT file {filename}")
        if backend == 'pd':
            sample = cls._read_tree_pandas_backend(tree)
        elif backend == 'ak':
            sample = cls._read_tree_awkward_backend(tree)
        else:
            raise ValueError("Only pd or ak backends are supported.")

        if metadata_hist is not None:
            logging.info("Getting metadata")
            metadata = file[metadata_hist].values()
            sample['metadata'] = tf.tile(tf.constant(metadata)[tf.newaxis, :], [sample['eventNumber'].shape[0], 1])

        logging.info(f'Done loading file:{filename}')
        dataset = tf.data.Dataset.from_tensor_slices(sample)
        return cls(dataset, list(sample.keys()))

    @classmethod
    def _read_tree_pandas_backend(cls, tree: uproot.TBranch) -> ROOTVariables:
        variables = tree.keys()
        sample = {}
        for var in variables:
            df = tree[var].array(library="ak")
            df: pd.DataFrame = ak.to_pandas(df)
            if df.empty:
                continue
            sample[var] = pandas_to_tensor(df['values'])
            logging.info(f'{var}: {sample[var].shape}')
        return sample

    @classmethod
    def _read_tree_awkward_backend(cls, tree: uproot.TBranch) -> ROOTVariables:
        variables = tree.keys()
        sample = {}
        for var in variables:
            df = tree[var].array(library="ak")
            if ak.size(ak.flatten(df, axis=None)) == 0:
                continue
            sample[var] = awkward_to_tensor(df)
            logging.info(f'{var}: {sample[var].shape}')
        return sample

    @classmethod
    def _concat(cls, datasets: List[ROOTDataset]) -> ROOTDataset:
        for dataset in datasets:
            if dataset.variables != datasets[0].variables:
                raise ValueError("Variables of datasets do not match")
        final_dataset = datasets[0]._dataset
        for ds in datasets[1:]:
            final_dataset = final_dataset.concatenate(ds._dataset)
        return cls(final_dataset, datasets[0].variables)

    @classmethod
    def from_root_files(cls, filenames: Union[List[str], str]) -> ROOTDataset:
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
