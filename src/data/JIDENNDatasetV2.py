from __future__ import annotations
import tensorflow as tf
from dataclasses import dataclass
from typing import Union, Literal, Callable
import os
import pickle
#
import src.config.config_subclasses as cfg
from .utils.CutV2 import Cut
from .utils.Expression import Expression
from .utils.functions import split_train_dev_test


ROOTVariables = dict[str, tf.RaggedTensor]
JIDENNVariables = dict[str, ROOTVariables]


@dataclass
class JIDENNDataset:
    variables: cfg.Variables
    target: Union[str, None] = None
    weight: Union[str, None] = None

    def load(self, file: str):
        element_spec_file = os.path.join(file, 'element_spec')
        with open(element_spec_file, 'rb') as f:
            element_spec = pickle.load(f)
        self._element_spec = element_spec
        self._datasets = [tf.data.experimental.load(
            file, compression='GZIP', element_spec=self._element_spec)]

    def split(self, test_size: float = 0.1, dev_size: float = 0.1):
        self._datasets = split_train_dev_test(self._datasets[0], test_size, dev_size)



    def process(self, cut: Union[Cut, None]):
        def _process(dataset):
            dataset = dataset.filter(cut) if cut is not None else dataset
            dataset = dataset.map(self._var_picker)
            return dataset
        self._datasets = [_process(dataset) for dataset in self._datasets]

    @property
    def train(self):
        return self._datasets[0]

    @property
    def dev(self):
        return self._datasets[1]

    @property
    def test(self):
        return self._datasets[2]

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
        self._datasets = [dataset.rejection_resample(
            label_func, target_dist=target_dist).map(lambda _, data: data) for dataset in self._datasets]
