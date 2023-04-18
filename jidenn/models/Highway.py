import tensorflow as tf
from typing import Callable, Union, Tuple


class HighwayModel(tf.keras.Model):
    def __init__(self,
                 layer_size: int,
                 num_layers: int,
                 input_size: Union[Tuple[int, int], int],
                 output_layer: tf.keras.layers.Layer,
                 activation: Callable,
                 dropout: Union[float, None] = None,
                 preprocess: Union[tf.keras.layers.Layer, None] = None) -> None:

        self._activation = activation
        self.layer_size, self.num_layers, self.dropout = layer_size, num_layers, dropout
        inputs = tf.keras.layers.Input(shape=(input_size, ))
        preprocessed = preprocess(inputs) if preprocess is not None else inputs

        hidden = self.gated_hidden_layers(preprocessed)

        output = output_layer(hidden)

        super().__init__(inputs=inputs, outputs=output)

    def gated_hidden_layers(self, inputs: tf.Tensor) -> tf.Tensor:
        hidden = tf.keras.layers.Dense(self.layer_size, activation=self._activation)(inputs)
        for _ in range(self.num_layers):
            pure_hidden = tf.keras.layers.Dense(self.layer_size, activation=self._activation)(hidden)
            gate = tf.keras.layers.Dense(self.layer_size, activation=tf.nn.sigmoid,
                                         bias_initializer=tf.keras.initializers.Constant(-2.0))(hidden)
            scaled_gate = tf.keras.layers.Multiply()([gate, hidden])
            scaled_pure_hidden = tf.keras.layers.Multiply()([1 - gate, pure_hidden])
            hidden = tf.keras.layers.Add()([scaled_gate, scaled_pure_hidden])
            if self.dropout is not None:
                hidden = tf.keras.layers.Dropout(self.dropout)(hidden)
        return hidden
