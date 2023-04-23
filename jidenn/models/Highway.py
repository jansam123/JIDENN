r"""
Implementation of the Highway Network model based on the paper https://arxiv.org/abs/1505.00387.

It is an extension of fully-connected networks, which allows the network to learn to skip layers.
The $i$-th layer is computed as (implicit einstein sumation convention)
$$ x^{(l+1)}_i = f(W^{(l)}_{ij} x^{(l)}_j + b^{(l)}_i) (1-p) + x^{(l)}_i p $$
where $p$ is the probability of skipping the layer. It is computed 
from the input as 
$$ p = \sigma (W^{(l)}_{ij, p} x_j + b_{i, p}) $$
with the sigmoid activation function $\sigma$ giving the probability of skipping the layer. 

The main advantage of the Highway Network is that it allows the network to learn to skip layers,
i.e. not having to learn the identity function or relearning the same function in each layer.

These Highway based models have a tendency to **not diverge** if the depth of the network is increased,
as may happen with the fully connected networks.

![Highway](/diagrams/highway.png)
"""

import tensorflow as tf
from typing import Callable, Optional


class HighwayModel(tf.keras.Model):
    """Implements a Highway Network model.

    The expected input shape is `(batch_size, input_size)`.

    The model already contains the `tf.keras.layers.Input` layer, so it can be used as a standalone model.

    Args:
        layer_size (int): The number of neurons in each hidden layer.
        num_layers (int): The number of hidden layers.
        input_size (int): The size of the input.
        output_layer (tf.keras.layers.Layer): The output layer of the model.
        activation (Callable[[tf.Tensor], tf.Tensor]) The activation function to use in the hidden layers.
        dropout (float, optional): The dropout rate to use in the hidden layers. Defaults to None.
        preprocess (tf.keras.layers.Layer, optional): The preprocessing layer to use. Defaults to None.
    """

    def __init__(self,
                 layer_size: int,
                 num_layers: int,
                 input_size: int,
                 output_layer: tf.keras.layers.Layer,
                 activation: Callable[[tf.Tensor], tf.Tensor],
                 dropout: Optional[float] = None,
                 preprocess: Optional[tf.keras.layers.Layer] = None) -> None:

        self._activation = activation
        self.layer_size, self.num_layers, self.dropout = layer_size, num_layers, dropout
        inputs = tf.keras.layers.Input(shape=(input_size, ))
        preprocessed = preprocess(inputs) if preprocess is not None else inputs

        hidden = self.gated_hidden_layers(preprocessed)

        output = output_layer(hidden)

        super().__init__(inputs=inputs, outputs=output)

    def gated_hidden_layers(self, inputs: tf.Tensor) -> tf.Tensor:
        """Computes the hidden layers of the model.

        Args:
            inputs (tf.Tensor): The input tensor of shape `(batch_size, input_size)`.

        Returns:
            tf.Tensor: The output tensor of shape `(batch_size, layer_size)`.
        """
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
