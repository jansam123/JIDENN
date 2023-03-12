import tensorflow as tf
import tensorflow_addons as tfa
from typing import Union, Tuple, Optional, Callable, List

from src.config import config_subclasses as cfg
from src.config import model_config as model_cfg
from .get_optimizer import get_optimizer

from ..models.BasicFCModel import BasicFCModel
from ..models.HighwayModel import HighwayModel
from ..models.TransformerModel import TransformerModel
from ..models.ParTModel import ParTModel
from ..models.DeParTModel import DeParTModel
from ..models.PFNModel import PFNModel
from ..models.BDT import get_BDT_model


class ModelBuilder:
    def __init__(self, args_model: model_cfg.ModelConfig, input_size: int, num_labels: int):
        self.args_model = args_model
        self.input_size = input_size
        self.num_labels = num_labels

    def metrics(self) -> List[tf.keras.metrics.Metric]:
        metrics = [tf.keras.metrics.CategoricalAccuracy() if self.num_labels > 2 else tf.keras.metrics.BinaryAccuracy(),
                   tf.keras.metrics.AUC()]
        return metrics

    def loss(self, label_smoothing: float) -> tf.keras.losses.Loss:
        if self.num_labels > 2:
            return tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
        else:
            return tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)

    def output_layer(self) -> tf.keras.layers.Layer:
        if self.num_labels > 2:
            return tf.keras.layers.Dense(self.num_labels, activation=tf.nn.softmax)
        else:
            return tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

    def activation(self, activation: str) -> Callable:
        if activation == 'relu':
            return tf.nn.relu
        elif activation == 'gelu':
            return tfa.activations.gelu
        elif activation == 'tanh':
            return tf.nn.tanh
        elif activation == 'swish':
            return tf.keras.activations.swish
        else:
            raise NotImplementedError(f'Activation {activation} not supported.')
