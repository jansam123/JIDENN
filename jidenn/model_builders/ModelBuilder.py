"""
Module for building models from config and compiling them.
"""

import tensorflow as tf
import keras
from typing import Union, Tuple, List, Literal
from dataclasses import dataclass

from jidenn.config import config
from .optimizer_initialization import get_optimizer
from ..evaluation.evaluation_metrics import EffectiveTaggingEfficiency

from .model_initialization import model_getter_lookup


@dataclass
class ModelBuilder:
    """Class for building models from config.
    Provides a facade between pure model initialization and model building with output layer, optimizer, loss, and metrics, with subsequent compilation.

    Args:
        model_name (str): Name of model to build. Options are 'fc', 'highway', 'pfn', 'efn', 'transformer', 'part', 'depart', 'bdt'
        args_model (config.Models): Model config
        input_size (Union[int, Tuple[None, int], Tuple[Tuple[None, int], Tuple[None, None, int]]]): Input size
        num_labels (int): Number of labels, i.e. size of output layer
        args_optimizer (config.Optimizer): Optimizer config
        preprocess (Union[keras.layers.Layer, None, Tuple[keras.layers.Layer, keras.layers.Layer]], optional): Preprocessing layer(s). Defaults to None.

    """
    model_name: Literal['fc', 'highway', 'pfn', 'efn', 'transformer', 'part', 'depart', 'bdt']
    args_model: config.Models
    input_size: Union[int, Tuple[None, int], Tuple[Tuple[None, int], Tuple[None, None, int]]]
    num_labels: int
    args_optimizer: config.Optimizer
    preprocess: Union[keras.layers.Layer, None, Tuple[keras.layers.Layer, keras.layers.Layer]] = None

    @property
    def model(self) -> keras.Model:
        """Builds model from config."""
        model_getter = model_getter_lookup(self.model_name)
        # get rid of number in model name
        stripped_model_name = ''.join([i for i in self.model_name if not i.isdigit()])
        model = model_getter(input_size=self.input_size,
                             output_layer=self.output_layer,
                             args_model=getattr(self.args_model, stripped_model_name),
                             preprocess=self.preprocess)
        return model

    @property
    def compiled_model(self) -> keras.Model:
        """Compiled Model."""
        model = self.model

        if self.model_name == 'bdt':
            model.compile(weighted_metrics=self.metrics)
            return model

        model.compile(optimizer=self.optimizer,
                      loss=self.loss,
                      weighted_metrics=self.metrics)
        return model

    @property
    def optimizer(self) -> keras.optimizers.Optimizer:
        """Instantiates optimizer from config."""
        return get_optimizer(self.args_optimizer)

    @property
    def metrics(self) -> List[keras.metrics.Metric]:
        """Metrics used in training."""
        metrics = [keras.metrics.CategoricalAccuracy() if self.num_labels >
                   2 else keras.metrics.BinaryAccuracy(),]
        return metrics

    @property
    def loss(self) -> keras.losses.Loss:
        """Loss function used in training."""""
        if self.num_labels > 2:
            return keras.losses.CategoricalCrossentropy(label_smoothing=self.args_optimizer.label_smoothing)
        else:
            return keras.losses.BinaryCrossentropy(label_smoothing=self.args_optimizer.label_smoothing)

    @property
    def output_layer(self) -> keras.layers.Layer:
        """Output layer for model. It is the same for all models."""
        if self.num_labels > 2:
            return keras.layers.Dense(self.num_labels, activation=keras.activations.softmax)
        else:
            return keras.layers.Dense(1, activation=keras.activations.sigmoid)

    