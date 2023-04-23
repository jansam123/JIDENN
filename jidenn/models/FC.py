"""
Implements a fully connected neural network, i.e. a multi-layer perceptron following the traditional notation,
a layer is a linear transformation followed by an activation function (implicit einstein sumation convention):
$$ x^{(l+1)}_i = f(W^{(l)}_{ij} x^{(l)}_j + b^{(l)}_i) $$
where $x^{(l)}_i$ is the $i$-th neuron in layer $l$, $W^{(l)}_{ij}$ is the weight between $i$-th neuron in layer $l$, 
f is the activation function, and $b^{(l)}_i$ is the bias of the $i$-th neuron in layer $l$.

![FC](/diagrams/fc.png)
"""
import tensorflow as tf
from typing import Callable, Optional


class FCModel(tf.keras.Model):
    """Implements a fully connected neural network, i.e. a multi-layer perceptron.

    The expected input shape is `(batch_size, input_size)`.

    The model already contains the `tf.keras.layers.Input` layer, so it can be used as a standalone model.


    Args:
        layer_size (int): The number of neurons in each hidden layer.
        num_layers (int): The number of hidden layers.
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
        hidden = self.hidden_layers(preprocessed)
        output = output_layer(hidden)
        super().__init__(inputs=inputs, outputs=output)

    def hidden_layers(self, inputs: tf.Tensor) -> tf.Tensor:
        """Executes the hidden layers of the model.

        Args:
            inputs (tf.Tensor): The input tensor of shape `(batch_size, input_size)`.

        Returns:
            tf.Tensor: The output tensor of shape `(batch_size, layer_size)`.
        """
        hidden = inputs
        for _ in range(self.num_layers):
            hidden = tf.keras.layers.Dense(self.layer_size, activation=self._activation)(hidden)
            hidden = tf.keras.layers.Dropout(self.dropout)(hidden)
        return hidden
