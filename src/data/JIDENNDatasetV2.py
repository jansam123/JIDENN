from __future__ import annotations
import tensorflow as tf
from dataclasses import dataclass
from typing import Union, Literal, Callable
import os
import pickle
#
from utils.CutV2 import Cut
import src.config.config_subclasses as cfg
from src.data.utils.Expression import Expression
# import pandas as pd
# import time
# import os@dataclass


ROOTVariables = dict[str, tf.RaggedTensor]
JIDENNVariables = dict[str, ROOTVariables]


@dataclass
class JIDENNDataset:
    file: str
    variables: cfg.Variables
    target: Union[str, None] = None
    weight: Union[str, None] = None
    cut: Union[Cut, None] = None

    def __post_init__(self):
        element_spec_file = os.path.join(self.file, 'element_spec')
        with open(element_spec_file, 'rb') as f:
            element_spec = pickle.load(f)
        self._element_spec = element_spec
        dataset = tf.data.experimental.load(self.file, compression='GZIP', element_spec=self._element_spec)
        dataset = dataset.filter(self.cut) if self.cut is not None else dataset
        dataset = dataset.map(self._pick_variables)
        self._dataset = dataset

    @property
    def dataset(self):
        return self._dataset

    @tf.function
    def _pick_variables(self, sample: ROOTVariables) -> Union[tuple[JIDENNVariables, tf.RaggedTensor, tf.RaggedTensor], JIDENNVariables, tuple[JIDENNVariables, tf.RaggedTensor, Literal[1]]]:
        new_sample = {'perEvent': {}, 'perJet': {}, 'perJetTuple': {}}
        for var in self.variables.perEvent:
            new_sample['perEvent'][var] = Expression(var)(sample)
        for var in self.variables.perJet:
            new_sample['perJet'][var] = Expression(var)(sample)
        for var in self.variables.perJetTuple:
            new_sample['perJetTuple'][var] = Expression(var)(sample)

        if self.target is None:
            return new_sample
        if self.weight is None:
            return new_sample, Expression(self.target)(sample), 1.
        else:
            return new_sample, Expression(self.target)(sample), Expression(self.weight)(sample)

    def resample_by_label(self, label_func: Callable[[JIDENNVariables, int, float], int], target_dist: list[float]):
        self._dataset = self._dataset.rejection_resample(label_func, target_dist=target_dist)
