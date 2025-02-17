"""
Implements a fully connected neural network, i.e. a multi-layer perceptron following the traditional notation,
a layer is a linear transformation followed by an activation function (implicit einstein sumation convention):
$$ x^{(l+1)}_i = f(W^{(l)}_{ij} x^{(l)}_j + b^{(l)}_i) $$
where $x^{(l)}_i$ is the $i$-th neuron in layer $l$, $W^{(l)}_{ij}$ is the weight between $i$-th neuron in layer $l$, 
f is the activation function, and $b^{(l)}_i$ is the bias of the $i$-th neuron in layer $l$.

![FC](images/fc.png)
"""
import tensorflow as tf
import keras
from typing import Callable, Optional


class FCModel(keras.Model):
    """Implements a fully connected neural network, i.e. a multi-layer perceptron.

    The expected input shape is `(batch_size, input_size)`.

    The model already contains the `keras.layers.Input` layer, so it can be used as a standalone model.


    Args:
        layer_size (int): The number of neurons in each hidden layer.
        num_layers (int): The number of hidden layers.
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
        hidden = self.hidden_layers(preprocessed)
        output = output_layer(hidden)
        super().__init__(inputs=inputs, outputs=output, **kwargs)

    def hidden_layers(self, inputs: tf.Tensor) -> tf.Tensor:
        """Executes the hidden layers of the model.

        Args:
            inputs (tf.Tensor): The input tensor of shape `(batch_size, input_size)`.

        Returns:
            tf.Tensor: The output tensor of shape `(batch_size, layer_size)`.
        """
        hidden = inputs
        for _ in range(self.num_layers):
            hidden = keras.layers.Dense(self.layer_size, activation=self._activation)(hidden)
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