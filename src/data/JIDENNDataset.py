from __future__ import annotations
from dataclasses import dataclass
import tensorflow as tf
from typing import Callable
import pickle
import os
import awkward as ak
import vector
import uproot
#
import src.config.config_subclasses as cfg
from .utils.Expression import Expression
# import pandas as pd
# import time
# import os



@dataclass
class JIDENNDataset:
    files: list[str]
    variables: cfg.Variables
    element_spec_file: str | None = None
    target: str | None = None
    weight: str | None = None
    cut: str | None = None
    filter: Callable | None = None

    def get_preprocess_mapping(self):
        @tf.function
        def pick_variables(sample):
            perJet = tf.stack([tf.cast(Expression(var)(sample), tf.float32) for var in self.variables.perJet], axis=-1)

            perEvent = tf.stack([tf.cast(Expression(var)(sample), tf.float32)
                                for var in self.variables.perEvent], axis=-1) if self.variables.perEvent is not None else None
            perEvent = tf.tile(perEvent[tf.newaxis, :], [tf.shape(perJet)[0], 1]) if perEvent is not None else None

            label = Expression(self.target)(sample) if self.target is not None else None
            weight = Expression(self.weight)(sample) if self.weight is not None else None

            weight = tf.fill([tf.shape(perJet)[0]], weight) if weight is not None else tf.ones_like(
                label, dtype=tf.float32)

            if self.variables.perJetTuple is not None:
                perJetTuple = tf.stack([Expression(var)(sample) for var in self.variables.perJetTuple], axis=-1)
                return (tf.concat([perJet, perEvent], axis=-1), perJetTuple), label, weight
            else:
                return tf.concat([perJet, perEvent], axis=-1), label, weight
        return pick_variables

    def load_single_dataset(self, file) -> tf.data.Dataset:
        dataset = tf.data.experimental.load(file, compression='GZIP', element_spec=self.dataset_TypeSpec)
        dataset = (dataset.map(self.get_preprocess_mapping(), num_parallel_calls=tf.data.AUTOTUNE)
                   .flat_map(lambda *x: tf.data.Dataset.from_tensor_slices(x)))
        dataset = dataset.filter(self.filter) if self.filter is not None else dataset
        return dataset

    @property
    def dataset(self):
        return tf.data.Dataset.from_tensor_slices(self.files).interleave(self.load_single_dataset, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

    @property
    def dataset_TypeSpec(self):
        spec_file = self.element_spec_file if self.element_spec_file is not None else self.files[0]
        with open(spec_file, 'rb') as in_:
            type_spec = pickle.load(in_)
        return type_spec
