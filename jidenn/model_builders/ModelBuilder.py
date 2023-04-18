import tensorflow as tf
from typing import Union, Tuple, Optional, Callable, List

from jidenn.config import config_subclasses as cfg
from .get_optimizer import get_optimizer

from .get_model import model_getter_lookup


class ModelBuilder:
    def __init__(self,
                 model_name: str,
                 args_model: cfg.Models,
                 input_size: Union[int, Tuple[None, int], Tuple[Tuple[None, int], Tuple[None, None, int]]],
                 num_labels: int,
                 args_optimizer: cfg.Optimizer,
                 preprocess: Union[tf.keras.layers.Layer, None, Tuple[tf.keras.layers.Layer, tf.keras.layers.Layer]] = None):

        self.preprocess = preprocess
        self.args_models = args_model
        self.args_optimizer = args_optimizer
        self.input_size = input_size
        self.num_labels = num_labels
        self.model_name = model_name

    @property
    def model(self) -> tf.keras.Model:
        model_getter = model_getter_lookup(self.model_name)
        model = model_getter(input_size=self.input_size,
                             output_layer=self.output_layer,
                             args_model=getattr(self.args_models, self.model_name),
                             preprocess=self.preprocess)
        return model

    @property
    def compiled_model(self) -> tf.keras.Model:
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
        return get_optimizer(self.args_optimizer)

    @property
    def metrics(self) -> List[tf.keras.metrics.Metric]:
        metrics = [tf.keras.metrics.CategoricalAccuracy() if self.num_labels > 2 else tf.keras.metrics.BinaryAccuracy(),
                   tf.keras.metrics.AUC()]
        return metrics

    @property
    def loss(self) -> tf.keras.losses.Loss:
        if self.num_labels > 2:
            return tf.keras.losses.CategoricalCrossentropy(label_smoothing=self.args_optimizer.label_smoothing)
        else:
            return tf.keras.losses.BinaryCrossentropy(label_smoothing=self.args_optimizer.label_smoothing)

    @property
    def output_layer(self) -> tf.keras.layers.Layer:
        if self.num_labels > 2:
            return tf.keras.layers.Dense(self.num_labels, activation=tf.nn.softmax)
        else:
            return tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
