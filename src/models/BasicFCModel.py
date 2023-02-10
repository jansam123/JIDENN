# from __future__ import annotations
import tensorflow as tf
from typing import Callable, Union


class BasicFCModel(tf.keras.Model):
    def __init__(self,
                 hidden_layer_size: int,
                 num_layers: int,
                 input_size: int,
                 output_layer: tf.keras.layers.Layer,
                 activation: Callable,
                 dropout: Union[float, None] = None,
                 preprocess: Union[tf.keras.layers.Layer, None] = None) -> None:

        self._activation = activation
        inputs = tf.keras.layers.Input(shape=(input_size, ))
        preprocessed = preprocess(inputs) if preprocess is not None else inputs
        hidden = self.hidden_layers(preprocessed, hidden_layer_size, num_layers, dropout)
        output = output_layer(hidden)
        super().__init__(inputs=inputs, outputs=output)

    def hidden_layers(self,
                       inputs: tf.Tensor,
                       layer_size: int,
                       num_layers: int,
                       dropout: Union[float, None] = None) -> tf.Tensor:
        hidden = inputs
        for _ in range(num_layers):
            hidden = tf.keras.layers.Dense(layer_size, activation=self._activation)(hidden)
            if dropout is not None:
                hidden = tf.keras.layers.Dropout(dropout)(hidden)
        return hidden
