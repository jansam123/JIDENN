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

![Highway](images/highway.png)
"""

import tensorflow as tf
import keras
from typing import Callable, Optional


class HighwayModel(keras.Model):
    """Implements a Highway Network model.

    The expected input shape is `(batch_size, input_size)`.

    The model already contains the `keras.layers.Input` layer, so it can be used as a standalone model.

    Args:
        layer_size (int): The number of neurons in each hidden layer.
        num_layers (int): The number of hidden layers.
        input_size (int): The size of the input.
        output_layer (keras.layers.Layer): The output layer of the model.
        activation (Callable[[tf.Tensor], tf.Tensor]) The activation function to use in the hidden layers.
        dropout (float, optional): The dropout rate to use in the hidden layers. Defaults to None.
        preprocess (keras.layers.Layer, optional): The preprocessing layer to use. Defaults to None.
    """

    def __init__(self,
                 layer_size: int,
                 num_layers: int,
                 input_size: int,
                 output_layer: keras.layers.Layer,
                 activation: Callable[[tf.Tensor], tf.Tensor],
                 dropout: Optional[float] = None,
                 preprocess: Optional[keras.layers.Layer] = None,
                 **kwargs) -> None:

        self._activation = activation
        self.layer_size, self.num_layers, self.dropout = layer_size, num_layers, dropout
        self.input_size = input_size
        self.output_layer = output_layer
        self.preprocess = preprocess
        
        inputs = keras.layers.Input(shape=(input_size, ))
        preprocessed = preprocess(inputs) if preprocess is not None else inputs

        hidden = self.gated_hidden_layers(preprocessed)

        output = output_layer(hidden)

        super().__init__(inputs=inputs, outputs=output, **kwargs)

    def gated_hidden_layers(self, inputs: tf.Tensor) -> tf.Tensor:
        """Computes the hidden layers of the model.

        Args:
            inputs (tf.Tensor): The input tensor of shape `(batch_size, input_size)`.

        Returns:
            tf.Tensor: The output tensor of shape `(batch_size, layer_size)`.
        """
        hidden = keras.layers.Dense(self.layer_size, activation=self._activation)(inputs)
        for _ in range(self.num_layers):
            pure_hidden = keras.layers.Dense(self.layer_size, activation=self._activation)(hidden)
            gate = keras.layers.Dense(self.layer_size, activation=keras.activations.sigmoid,
                                         bias_initializer=keras.initializers.Constant(-2.0))(hidden)
            scaled_gate = keras.layers.Multiply()([gate, hidden])
            scaled_pure_hidden = keras.layers.Multiply()([1 - gate, pure_hidden])
            hidden = keras.layers.Add()([scaled_gate, scaled_pure_hidden])
            if self.dropout is not None:
                hidden = keras.layers.Dropout(self.dropout)(hidden)
        return hidden

    def get_config(self):
        config = super().get_config()
        config.update({
            'layer_size': self.layer_size,
            'num_layers': self.num_layers,
            'output_layer': keras.saving.serialize_keras_object(self.output_layer),
            'activation': keras.activations.serialize(self._activation),
            'dropout': self.dropout,
            'preprocess': keras.saving.serialize_keras_object(self.preprocess),
            'input_size': self.input_size
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        if config.get('output_layer') is not None:
            config['output_layer'] = keras.saving.deserialize_keras_object(config['output_layer'])
        if config.get('preprocess') is not None:
            config['preprocess'] = keras.saving.deserialize_keras_object(config['preprocess'])
        config['activation'] = keras.activations.deserialize(config['activation'])
        return cls(**config)