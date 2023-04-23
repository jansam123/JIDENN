r"""
Implementation of the Particle Flow Network (PFN) model.
See https://energyflow.network for original implementation or
the paper https://arxiv.org/abs/1810.05165 for more details.

The input are the jet constituents, i.e. the particles in the jet foorming an input shape of `(batch_size, num_particles, num_features)`.
The model is then executed as follows:
$$ \mathrm{output} = F\left(\sum_i  \Phi{(\eta_i,\phi_i, \pmb{E_i})}\right) $$

The F function is any function, which is applied to the summed features. The default is a fully-connected network.
The mapping Phi can be a fully-connected network or a convolutional network. The default is a fully-connected network.

The PFN is a modification of the EFN, where the energy part is also mapped with the Phi function.
However the model then **violates Infrared and Collinear safety**.

![EFN_PFN](../../../diagrams/pfn_efn.png)
On the left is the PFN model, on the right the EFN model.
"""

import tensorflow as tf
from typing import Optional, List, Literal, Tuple, Callable


class PFNModel(tf.keras.Model):
    """Implements the Particle Flow Network (PFN) model.

    The expected input shape is `(batch_size, num_particles, num_features)`, where the second dimension is ragged.

    Args:
        input_shape (Tuple[None, int]): The shape of the input.
        Phi_sizes (List[int]): The sizes of the hidden layers of the Phi function.
        F_sizes (List[int]): The sizes of the hidden layers of the F function.
        output_layer (tf.keras.layers.Layer): The output layer of the model.
        activation (Callable[[tf.Tensor], tf.Tensor]) The activation function. 
        Phi_backbone (str, optional): The backbone of the Phi function. Options are "fc" for a fully-connected network 
            and "cnn" for a convolutional network. Defaults to "fc".
            This option is omitted from the `jidenn.config.model_config.PFN` as it is not used in the paper.
        batch_norm (bool, optional): Whether to use batch normalization. Defaults to False.
            This option is omitted from the `jidenn.config.model_config.PFN` as it is not used in the paper.
        Phi_dropout (float, optional): The dropout rate of the Phi function. Defaults to None.
        F_dropout (float, optional): The dropout rate of the F function. Defaults to None.
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
        row_lengths = input.row_lengths()
        mask = tf.sequence_mask(row_lengths)
        hidden = input.to_tensor()

        if preprocess is not None:
            hidden = preprocess(hidden)

        if batch_norm:
            hidden = tf.keras.layers.BatchNormalization()(hidden)

        if Phi_backbone == "cnn":
            hidden = self.cnn_Phi(hidden)
        elif Phi_backbone == "fc":
            hidden = self.fc_Phi(hidden)
        else:
            raise ValueError(f"backbone must be either 'cnn' or 'fc', not {Phi_backbone}")

        hidden = hidden * tf.expand_dims(tf.cast(mask, tf.float32), -1)
        hidden = tf.math.reduce_sum(hidden, axis=1)
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
