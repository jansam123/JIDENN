# from __future__ import annotations
import tensorflow as tf
from typing import Callable, Union


class FCModel(tf.keras.Model):

    def __init__(self,
                 layer_size: int,
                 num_layers: int,
                 input_size: int,
                 output_layer: tf.keras.layers.Layer,
                 activation: Callable,
                 dropout: Union[float, None] = None,
                 preprocess: Union[tf.keras.layers.Layer, None] = None) -> None:

        self._activation = activation
        self.layer_size, self.num_layers, self.dropout = layer_size, num_layers, dropout

        inputs = tf.keras.layers.Input(shape=(input_size, ))
        preprocessed = preprocess(inputs) if preprocess is not None else inputs
        hidden = self.hidden_layers(preprocessed)
        output = output_layer(hidden)
        super().__init__(inputs=inputs, outputs=output)

    def hidden_layers(self, inputs: tf.Tensor) -> tf.Tensor:
        hidden = inputs
        for _ in range(self.num_layers):
            hidden = tf.keras.layers.Dense(self.layer_size, activation=self._activation)(hidden)
            if self.dropout is not None and self.dropout > 0:
                hidden = tf.keras.layers.Dropout(self.dropout)(hidden)
        return hidden
