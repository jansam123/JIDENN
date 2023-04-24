r"""
Module implementing the Energy Flow Network (EFN) model.
See https://energyflow.network for original implementation or
the paper https://arxiv.org/abs/1810.05165 for more details.

The input are the jet consituents, i.e. the particles in the jet foorming an input shape of `(batch_size, num_particles, num_features)`.
The angular features are mapped with the Phi mapping, afterwards the energy variables are multiplied by the angular features 
of each constituent separately and summed up. A F function is applied to the summed features forming an output
$$ \mathrm{output} = F\left(\sum_i \pmb{E_i} \Phi{(\eta_i,\phi_i)}\right) $$

The F function is any function, which is applied to the summed features. The default is a fully-connected network.
The mapping Phi can be a fully-connected network or a convolutional network. The default is a fully-connected network.

The energy part of the input is not used in the model to maintain **Infrared and Collinear safety** of the model.

![EFN_PFN](images/pfn_efn.png)
On the left is the PFN model, on the right the EFN model.
"""
import tensorflow as tf
from typing import Optional, Callable, List, Literal, Tuple


class EinsumLayer(tf.keras.layers.Layer):
    """
    This is needed to wrap the einsum operation, because the einsum operation produces an error when loded from a saved model with tf.keras.models.load_model.
    For more information see https://github.com/keras-team/keras/issues/15783.
    For more information about the einsum operation see https://www.tensorflow.org/api_docs/python/tf/einsum.

    Example:
    ```python
    x = EinsumLayer("bmhwf,bmoh->bmowf")((x1, x2))
    ```
    Args:
        equation (str): The equation to be used in the einsum operation.
    """

    def __init__(self, equation: str):
        super().__init__()
        self.equation = equation

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Call the layer.

        Args:
            inputs (Tuple[tf.Tensor, tf.Tensor]): The inputs to the layer with shapes described in the equation.

        Returns:
            tf.Tensor: The output of the layer.
        """
        return tf.einsum(self.equation, *inputs)

    def get_config(self):
        return {"equation": self.equation}


class EFNModel(tf.keras.Model):
    """The Energy Flow Network model.

    The input is expected to be a tensor of shape `(batch_size, num_particles, num_features=8)`,
    where the last 3 features are angular and the first 5 features are energy based.
    The second dimension is ragged, as the number of particles in each jet is not the same.
    See `jidenn.data.TrainInput.ConstituentVariables` for more details. 

    The model already contains the `tf.keras.layers.Input` layer, so it can be used as a standalone model.

    Args:
        input_shape (Tuple[None, int]): The shape of the input.
        Phi_sizes (List[int]): The sizes of the Phi layers.
        F_sizes (List[int]): The sizes of the F layers.
        output_layer (tf.keras.layers.Layer): The output layer.
        activation (Callable[[tf.Tensor], tf.Tensor]) The activation function for the Phi and F layers. 
        Phi_backbone (str, optional): The backbone of the Phi mapping. Options are "cnn" or "fc". Defaults to "fc".
            This argument is not in the config option, as the CNN backbone is not used in the paper and 
            might violate the Infrared and Collinear safety of the model. 
        batch_norm (bool, optional): Whether to use batch normalization. Defaults to False.
            This argument is not in the config option, as it is not used in the paper and 
            might violate the Infrared and Collinear safety of the model.
        F_dropout (float, optional): The dropout rate for the F layers. Defaults to None.
        preprocess (tf.keras.layers.Layer, optional): The preprocessing layer. Defaults to None.
    """

    def __init__(self,
                 input_shape: Tuple[None, int],
                 Phi_sizes: List[int],
                 F_sizes: List[int],
                 output_layer: tf.keras.layers.Layer,
                 activation: Callable[[tf.Tensor], tf.Tensor],
                 Phi_backbone: Literal["cnn", "fc"] = "fc",
                 batch_norm: bool = False,
                 Phi_dropout: Optional[float] = None,
                 F_dropout: Optional[float] = None,
                 preprocess: Optional[tf.keras.layers.Layer] = None):

        self.Phi_sizes, self.F_sizes = Phi_sizes, F_sizes
        self.Phi_dropout = Phi_dropout
        self.F_dropout = F_dropout
        self.activation = activation

        input = tf.keras.layers.Input(shape=input_shape, ragged=True)

        if preprocess is not None:
            input = preprocess(input)

        angular = input[:, :, 6:]
        energy = input[:, :, :6].to_tensor()

        row_lengths = angular.row_lengths()
        mask = tf.sequence_mask(row_lengths)
        angular = angular.to_tensor()

        if batch_norm:
            angular = tf.keras.layers.BatchNormalization()(angular)

        if Phi_backbone == "cnn":
            angular = self.cnn_Phi(angular)
        elif Phi_backbone == "fc":
            angular = self.fc_Phi(angular)
        else:
            raise ValueError(f"backbone must be either 'cnn' or 'fc', not {Phi_backbone}")

        angular = angular * tf.expand_dims(tf.cast(mask, tf.float32), -1)
        hidden = EinsumLayer('BPC,BPD->BCD')((angular, energy))
        hidden = tf.keras.layers.Flatten()(hidden)
        hidden = self.fc_F(hidden)
        output = output_layer(hidden)

        super().__init__(inputs=input, outputs=output)

    def cnn_Phi(self, inputs: tf.Tensor) -> tf.Tensor:
        """Convolutional Phi mapping.

        Args:
            inputs (tf.Tensor): The input tensor of shape `(batch_size, num_particles, num_features)`.

        Returns:
            tf.Tensor: The output tensor of shape `(batch_size, num_particles, Phi_sizes[-1])`.
        """
        hidden = inputs
        for size in self.Phi_sizes:
            hidden = tf.keras.layers.Conv1D(size, 1)(hidden)
            hidden = tf.keras.layers.BatchNormalization()(hidden)
            hidden = tf.keras.layers.Activation(self.activation)(hidden)
            if self.Phi_dropout is not None:
                hidden = tf.keras.layers.Dropout(self.Phi_dropout)(hidden)
        return hidden

    def fc_Phi(self, inputs: tf.Tensor) -> tf.Tensor:
        """Fully connected Phi mapping.

        Args:
            inputs (tf.Tensor): The input tensor of shape `(batch_size, num_particles, num_features)`.

        Returns:
            tf.Tensor: The output tensor of shape `(batch_size, num_particles, Phi_sizes[-1])`.
        """
        hidden = inputs
        for size in self.Phi_sizes:
            hidden = tf.keras.layers.Dense(size)(hidden)
            hidden = tf.keras.layers.Activation(self.activation)(hidden)
            if self.Phi_dropout is not None:
                hidden = tf.keras.layers.Dropout(self.Phi_dropout)(hidden)
        return hidden

    def fc_F(self, inputs: tf.Tensor) -> tf.Tensor:
        """Fully connected F mapping.

        Args:
            inputs (tf.Tensor): The input tensor of shape `(batch_size, num_features)`.

        Returns:
            tf.Tensor: The output tensor of shape `(batch_size, F_sizes[-1])`.
        """
        hidden = inputs
        for size in self.F_sizes:
            hidden = tf.keras.layers.Dense(size)(hidden)
            hidden = tf.keras.layers.Activation(self.activation)(hidden)
            if self.F_dropout is not None:
                hidden = tf.keras.layers.Dropout(self.F_dropout)(hidden)
        return hidden
