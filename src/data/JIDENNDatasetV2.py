from __future__ import annotations
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Union, Literal, Callable
import os
import pickle
#
import src.config.config_subclasses as cfg
from .utils.CutV2 import Cut
from .utils.Expression import Expression


ROOTVariables = dict[str, tf.RaggedTensor]
JIDENNVariables = dict[str, ROOTVariables]


@tf.function
def get_constituents(sample, label, weight):
    PFO_E = tf.math.sqrt(sample['perJetTuple']['jets_PFO_pt']**2 + sample['perJetTuple']['jets_PFO_m']**2)
    jet_E = tf.math.sqrt(sample['perJet']['jets_pt']**2 + sample['perJet']['jets_m']**2)
    deltaEta = sample['perJetTuple']['jets_PFO_eta'] - sample['perJet']['jets_eta']
    deltaPhi = sample['perJetTuple']['jets_PFO_phi'] - sample['perJet']['jets_phi']
    deltaR = tf.math.sqrt(deltaEta**2 + deltaPhi**2)

    logPT = tf.math.log(sample['perJetTuple']['jets_PFO_pt'])

    logPT_PTjet = tf.math.log(sample['perJetTuple']['jets_PFO_pt']/sample['perJet']['jets_pt'])
    logE = tf.math.log(PFO_E)
    logE_Ejet = tf.math.log(PFO_E/jet_E)
    m = sample['perJetTuple']['jets_PFO_m']
    data = [logPT, logPT_PTjet, logE, logE_Ejet, m, deltaEta, deltaPhi, deltaR]

    data = tf.stack(data, axis=-1)
    data = tf.cast(data, tf.float32)
    return data, label, weight

@tf.function
def get_high_level_variables(sample, label, weight):
    data = [tf.cast(sample['perJet'][var], tf.float32) for var in sample['perJet'].keys()]
    data += [tf.cast(sample['perEvent'][var], tf.float32) for var in sample['perEvent'].keys()]
    data = tf.stack(data, axis=-1)
    return data, label, weight

@dataclass
class JIDENNDataset:
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

    def load_element_spec(self, file: str) -> JIDENNDataset:
        with open(file, 'rb') as f:
            element_spec = pickle.load(f)
        return self._set_element_spec(element_spec)

    def process(self, cut: Union[Cut, None]) -> JIDENNDataset:
        if self.dataset is None:
            raise ValueError('Dataset not loaded yet.')
        dataset = self.dataset.filter(cut) if cut is not None else self.dataset
        dataset = dataset.map(self._var_picker)
        return self._set_dataset(dataset)

    def remap_labels(self, label_mapping: Callable[[int], int]) -> JIDENNDataset:
        if self.dataset is None:
            raise ValueError('Dataset not loaded yet.')

        @tf.function
        def remap_label(x, y, w):
            return x, label_mapping(y), w
        dataset = self.dataset.map(remap_label)
        return self._set_dataset(dataset)

    @property
    def dataset(self):
        return self._dataset

    @property
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

    @property
    def _var_picker(self):
        @tf.function
        def _pick_variables(sample: ROOTVariables) -> Union[tuple[JIDENNVariables, tf.RaggedTensor, tf.RaggedTensor], JIDENNVariables, tuple[JIDENNVariables, tf.RaggedTensor, Literal[1]]]:
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

    def resample_by_label(self, label_func: Callable[[JIDENNVariables, int, float], int], target_dist: list[float]):
        if self.dataset is None:
            raise ValueError('Dataset not loaded yet.')
        dataset = self.dataset.rejection_resample(label_func, target_dist=target_dist).map(lambda _, data: data)
        return self._set_dataset(dataset)

    @staticmethod
    def combine(datasets: list[JIDENNDataset], weights: list[float]) -> JIDENNDataset:
        dataset = tf.data.Dataset.sample_from_datasets([dataset.dataset for dataset in datasets], weights=weights)
        jidenn_dataset = JIDENNDataset(datasets[0].variables, datasets[0].target, datasets[0].weight)
        return jidenn_dataset._set_dataset(dataset)

    def apply(self, func: Callable[[tf.data.Dataset], tf.data.Dataset]) -> JIDENNDataset:
        if self.dataset is None:
            raise ValueError('Dataset not loaded yet.')
        dataset = func(self.dataset)
        return self._set_dataset(dataset)

    def to_pandas(self)-> pd.DataFrame:
        if self.dataset is None:
            raise ValueError('Dataset not loaded yet.')

        @tf.function
        def tuple_to_dict(data, label, weight):
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
                    map_func: Union[Callable[[JIDENNVariables, int, float], tuple[Union[tf.RaggedTensor, tf.Tensor], int, float]], None] = None) -> tf.data.Dataset:
        
        if self.dataset is None:
            raise ValueError('Dataset not loaded yet.')
        dataset = self.dataset.map(map_func) if map_func is not None else self.dataset
        dataset = dataset.shuffle(shuffle_buffer_size) if shuffle_buffer_size is not None else dataset
        if take is not None:
            dataset = dataset.take(take)
            dataset = dataset.apply(tf.data.experimental.assert_cardinality(take)) if assert_shape else dataset
        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(batch_size))
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

