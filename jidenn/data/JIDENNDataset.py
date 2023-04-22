from __future__ import annotations
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Union, Literal, Callable, Dict, Tuple, List, Optional
import os
import pickle
#
import jidenn.config.config_subclasses as cfg
from .utils.Cut import Cut
from .utils.Expression import Expression


ROOTVariables = Dict[str, Union[tf.RaggedTensor, tf.Tensor]]
JIDENNVariables = Dict[str, ROOTVariables]


@tf.function
def dict_to_stacked_array(data: Union[ROOTVariables, Tuple[ROOTVariables, ROOTVariables]], label: int, weight: Optional[float] = None) -> Tuple[Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]], int, Union[float, None]]:
    if isinstance(data, tuple):
        interaction = tf.stack([data[1][var] for var in data[1]], axis=-1)
        interaction = tf.where(tf.math.logical_or(tf.math.is_inf(interaction), tf.math.is_nan(interaction)),
                               tf.zeros_like(interaction), interaction)
        if weight is None:
            return (tf.stack([data[0][var] for var in data[0]], axis=-1), interaction), label
        return (tf.stack([data[0][var] for var in data[0]], axis=-1), interaction), label, weight
    else:
        if weight is None:
            return tf.stack([data[var] for var in data.keys()], axis=-1), label
        return tf.stack([data[var] for var in data.keys()], axis=-1), label, weight


@dataclass
class JIDENNDataset:
    """The JIDENNDataset class is a wrapper for a TensorFlow dataset that allows for easy loading and processing of JIDENN dataset files.

    Attributes:
        variables: A cfg.Variables object specifying the input variables to be used.
        target: The name of the target variable, or None if no target variable is specified.
        weight: The name of the weight variable, or None if no weight variable is specified.

    Methods:
        load_dataset(file: str) -> JIDENNDataset: Loads a JIDENN dataset from the specified file.

        save_dataset(file: str, num_shards: int) -> None: Saves the dataset to the specified file, with the specified number of shards.

        load_element_spec(file: str) -> JIDENNDataset: Loads the element spec for the dataset from the specified file.

        process(cut: Union[Cut, None]) -> JIDENNDataset: Processes the dataset using the specified cut.

        remap_labels(label_mapping: Callable[[int], int]) -> JIDENNDataset: Remaps the labels in the dataset using the specified label mapping function.

        resample_by_label(label_func: Callable[[JIDENNVariables, int, float], int], target_dist: List[float]) -> JIDENNDataset: Resamples the dataset by label using the specified label function and target distribution.

        combine(datasets: List[JIDENNDataset], weights: List[float]) -> JIDENNDataset: Combines the specified datasets with the specified weights.

        apply(func: Callable[[tf.data.Dataset], tf.data.Dataset]) -> JIDENNDataset: Applies the specified function to the dataset.

        to_pandas() -> pd.DataFrame: Converts the dataset to a Pandas dataframe.

        filter(filter: Callable) -> JIDENNDataset: Filters the dataset using the specified filter function.

        get_dataset(batch_size: int,  shuffle_buffer_size: Union[int, None] = None, assert_shape: bool = False, take: Union[int, None] = None, map_func: Union[Callable[[JIDENNVariables, int, float], Tuple[Union[tf.RaggedTensor, tf.Tensor], int, float]], None] = None) -> tf.data.Dataset:
        Returns a TensorFlow dataset with the specified parameters.
    """
    variables: cfg.Variables
    target: Union[str, None] = None
    weight: Union[str, None] = None

    def __post_init__(self):
        self._dataset = None
        self._element_spec = None

    def load_dataset(self, file: str) -> JIDENNDataset:
        if self.element_spec is None:
            element_spec_file = os.path.join(file, 'element_spec')
            jidenn_dataset = self.load_element_spec(element_spec_file)
        else:
            jidenn_dataset = self
        dataset = tf.data.experimental.load(
            file, compression='GZIP', element_spec=self.element_spec)
        return jidenn_dataset._set_dataset(dataset)

    def save_dataset(self, file: str, num_shards: int) -> None:
        if self.dataset is None:
            raise ValueError('Dataset not loaded yet.')

        @tf.function
        def random_shards(_) -> tf.Tensor:
            return tf.random.uniform(shape=[], minval=0, maxval=num_shards, dtype=tf.int64)

        tf.data.experimental.save(self.dataset, file, compression='GZIP', shard_func=random_shards)
        with open(os.path.join(file, 'element_spec'), 'wb') as f:
            pickle.dump(self.dataset.element_spec, f)

    def load_element_spec(self, file: str) -> JIDENNDataset:
        with open(file, 'rb') as f:
            element_spec = pickle.load(f)
        return self._set_element_spec(element_spec)

    def process(self, cut: Union[Cut, None]) -> JIDENNDataset:
        if self.dataset is None:
            raise ValueError('Dataset not loaded yet.')
        dataset = self.dataset.map(self._count_PFO())
        dataset = dataset.filter(cut) if cut is not None else dataset
        dataset = dataset.map(self._var_picker)
        return self._set_dataset(dataset)

    def remap_labels(self, label_mapping: Callable[[int], int]) -> JIDENNDataset:
        if self.dataset is None:
            raise ValueError('Dataset not loaded yet.')

        @tf.function
        def remap_label(x, y, w=None):
            return x, label_mapping(y)
        dataset = self.dataset.map(remap_label)
        return self._set_dataset(dataset)

    @ property
    def dataset(self):
        return self._dataset

    @ property
    def element_spec(self):
        return self._element_spec

    def _set_element_spec(self, element_spec: tf.TensorSpec) -> JIDENNDataset:
        jidenn_dataset = JIDENNDataset(variables=self.variables,
                                       target=self.target,
                                       weight=self.weight)
        self._element_spec = element_spec
        self._dataset = self.dataset
        return jidenn_dataset

    def _set_dataset(self, dataset: Union[tf.data.Dataset, None]) -> JIDENNDataset:
        jidenn_dataset = JIDENNDataset(variables=self.variables,
                                       target=self.target,
                                       weight=self.weight)
        jidenn_dataset._dataset = dataset
        jidenn_dataset._element_spec = dataset.element_spec
        return jidenn_dataset

    def _count_PFO(self, PFO_variables=['jets_PFO_pt', 'jets_PFO_eta', 'jets_PFO_phi', 'jets_PFO_m']):
        @tf.function
        def _wrapped_count_PFO(sample: ROOTVariables) -> ROOTVariables:
            for pfo_var in PFO_variables:
                if pfo_var in sample.keys():
                    sample = sample.copy()
                    sample['jets_PFO_n'] = tf.reduce_sum(tf.ones_like(sample['jets_PFO_pt']))
                    break
            return sample
        return _wrapped_count_PFO

    @ property
    def _var_picker(self):
        @tf.function
        def _pick_variables(sample: ROOTVariables) -> Union[Tuple[JIDENNVariables, tf.RaggedTensor, tf.RaggedTensor], JIDENNVariables, Tuple[JIDENNVariables, tf.RaggedTensor, Literal[1]]]:
            new_sample = {'perEvent': {}, 'perJet': {}, 'perJetTuple': {}}
            for var in self.variables.perEvent if self.variables.perEvent is not None else []:
                new_sample['perEvent'][var] = Expression(var)(sample)
            for var in self.variables.perJet if self.variables.perJet is not None else []:
                new_sample['perJet'][var] = Expression(var)(sample)
            for var in self.variables.perJetTuple if self.variables.perJetTuple is not None else []:
                new_sample['perJetTuple'][var] = Expression(var)(sample)

            if self.target is None:
                return new_sample
            if self.weight is None:
                return new_sample, Expression(self.target)(sample), 1.
            else:
                return new_sample, Expression(self.target)(sample), Expression(self.weight)(sample)
        return _pick_variables

    def resample_dataset(self, resampling_func: Callable[[JIDENNVariables, int, float], int], target_dist: List[float]):
        if self.dataset is None:
            raise ValueError('Dataset not loaded yet.')
        dataset = self.dataset.rejection_resample(resampling_func, target_dist=target_dist).map(lambda _, data: data)
        return self._set_dataset(dataset)

    @staticmethod
    def combine(datasets: List[JIDENNDataset], weights: List[float]) -> JIDENNDataset:
        dataset = tf.data.Dataset.sample_from_datasets([dataset.dataset for dataset in datasets], weights=weights)
        jidenn_dataset = JIDENNDataset(datasets[0].variables, datasets[0].target, datasets[0].weight)
        return jidenn_dataset._set_dataset(dataset)

    def apply(self, func: Callable[[tf.data.Dataset], tf.data.Dataset]) -> JIDENNDataset:
        if self.dataset is None:
            raise ValueError('Dataset not loaded yet.')
        dataset = func(self.dataset)
        return self._set_dataset(dataset)

    def map_data(self, func: Callable[[JIDENNVariables], Union[ROOTVariables, Tuple[ROOTVariables, ROOTVariables]]]) -> JIDENNDataset:
        if self.dataset is None:
            raise ValueError('Dataset not loaded yet.')

        @tf.function
        def input_wrapper(data, label, w=None):
            return func(data), label
        dataset = self.dataset.map(input_wrapper)
        return self._set_dataset(dataset)

    def to_pandas(self) -> pd.DataFrame:
        if self.dataset is None:
            raise ValueError('Dataset not loaded yet.')

        @tf.function
        def tuple_to_dict(data, label, weight=None):
            if isinstance(data, tuple):
                data = {**data[0], **data[1]}
            return {**data, 'label': label, 'weight': weight}

        dataset = self.dataset.map(tuple_to_dict)
        df = tfds.as_dataframe(dataset)
        df = df.rename(lambda x: x.replace('/', '.'), axis='columns')
        return df

    def filter(self, filter: Callable) -> JIDENNDataset:
        if self.dataset is None:
            raise ValueError('Dataset not loaded yet.')
        dataset = self.dataset.filter(filter)
        return self._set_dataset(dataset)

    def get_dataset(self, batch_size: int,
                    shuffle_buffer_size: Union[int, None] = None,
                    assert_shape: bool = False,
                    take: Union[int, None] = None,
                    map_func: Union[Callable[[Union[ROOTVariables, Tuple[ROOTVariables, ROOTVariables]], int, float], Tuple[ROOTVariables, int, float]], None] = None) -> tf.data.Dataset:

        if self.dataset is None:
            raise ValueError('Dataset not loaded yet.')
        if map_func is not None:
            dataset = self.dataset.map(map_func)
        else:
            dataset = self.dataset.map(dict_to_stacked_array)
        dataset = dataset.shuffle(shuffle_buffer_size) if shuffle_buffer_size is not None else dataset
        if take is not None:
            dataset = dataset.take(take)
            dataset = dataset.apply(tf.data.experimental.assert_cardinality(take)) if assert_shape else dataset
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(batch_size))
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
