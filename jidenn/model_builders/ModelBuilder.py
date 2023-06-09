"""
Module for building models from config and compiling them.
"""

import tensorflow as tf
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
        preprocess (Union[tf.keras.layers.Layer, None, Tuple[tf.keras.layers.Layer, tf.keras.layers.Layer]], optional): Preprocessing layer(s). Defaults to None.

    """
    model_name: Literal['fc', 'highway', 'pfn', 'efn', 'transformer', 'part', 'depart', 'bdt']
    args_model: config.Models
    input_size: Union[int, Tuple[None, int], Tuple[Tuple[None, int], Tuple[None, None, int]]]
    num_labels: int
    args_optimizer: config.Optimizer
    preprocess: Union[tf.keras.layers.Layer, None, Tuple[tf.keras.layers.Layer, tf.keras.layers.Layer]] = None

    @property
    def model(self) -> tf.keras.Model:
        """Builds model from config."""
        model_getter = model_getter_lookup(self.model_name)
        model = model_getter(input_size=self.input_size,
                             output_layer=self.output_layer,
                             args_model=getattr(self.args_model, self.model_name),
                             preprocess=self.preprocess)
        return model

    @property
    def compiled_model(self) -> tf.keras.Model:
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
    def optimizer(self) -> tf.keras.optimizers.Optimizer:
        """Instantiates optimizer from config."""
        return get_optimizer(self.args_optimizer)

    @property
    def metrics(self) -> List[tf.keras.metrics.Metric]:
        """Metrics used in training."""
        metrics = [tf.keras.metrics.CategoricalAccuracy() if self.num_labels >
                   2 else tf.keras.metrics.BinaryAccuracy(),]
        #    tf.keras.metrics.AUC(),
        #    EffectiveTaggingEfficiency()]
        return metrics

    @property
    def loss(self) -> tf.keras.losses.Loss:
        """Loss function used in training."""""
        if self.num_labels > 2:
            return tf.keras.losses.CategoricalCrossentropy(label_smoothing=self.args_optimizer.label_smoothing)
        else:
            return tf.keras.losses.BinaryCrossentropy(label_smoothing=self.args_optimizer.label_smoothing)

    @property
    def output_layer(self) -> tf.keras.layers.Layer:
        """Output layer for model. It is the same for all models."""
        if self.num_labels > 2:
            return tf.keras.layers.Dense(self.num_labels, activation=tf.nn.softmax)
        else:
            return tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
