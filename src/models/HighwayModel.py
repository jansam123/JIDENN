# from __future__ import annotations
import tensorflow as tf
from typing import Callable, Union, Tuple


class HighwayModel(tf.keras.Model):
    def __init__(self,
                 layer_size: int,
                 num_layers: int,
                 input_size: Union[Tuple[int, int], int],
                 output_layer: tf.keras.layers.Layer,
                 activation: Callable,
                 gate_activation: Callable = tf.nn.sigmoid,
                 dropout: Union[float, None] = None,
                 preprocess: Union[tf.keras.layers.Layer, None] = None) -> None:

        self._activation = activation
        self._gate_activation = gate_activation
        inputs = tf.keras.layers.Input(shape=(input_size, ))
        preprocessed = preprocess(inputs) if preprocess is not None else inputs

        hidden = self._highway_layers(preprocessed, layer_size, num_layers, dropout)

        output = output_layer(hidden)

        super().__init__(inputs=inputs, outputs=output)

    def _highway_layers(self, inputs, layer_size: int, num_layers: int, dropout: Union[float, None] = None) -> tf.Tensor:
        hidden = tf.keras.layers.Dense(layer_size, activation=self._activation)(inputs)
        for _ in range(num_layers):
            h = tf.keras.layers.Dense(layer_size, activation=self._activation)(hidden)
            t = tf.keras.layers.Dense(layer_size, activation=self._gate_activation,
                                      bias_initializer=tf.keras.initializers.Constant(-2.0))(hidden)
            hidden = tf.keras.layers.Add()([tf.keras.layers.Multiply()(
                [t, h]), tf.keras.layers.Multiply()([1-t, hidden])])
            if dropout is not None:
                hidden = tf.keras.layers.Dropout(dropout)(hidden)
        return hidden
