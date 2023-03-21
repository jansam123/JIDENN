import tensorflow as tf
from typing import Optional, Callable, List, Literal, Tuple


class EinsumLayer(tf.keras.layers.Layer):
    """
    This is needed to wrap the einsum operation, because the einsum operation produces an error when loded from a saved model with tf.keras.models.load_model.
    For more information see https://github.com/keras-team/keras/issues/15783 

    Usage:
    x = EinsumLayer("bmhwf,bmoh->bmowf")((x1, x2))
    """

    def __init__(self, equation: str):
        super().__init__()
        self.equation = equation

    def call(self, inputs, *args, **kwargs):
        return tf.einsum(self.equation, *inputs)

    def get_config(self):
        return {"equation": self.equation}


class EFNModel(tf.keras.Model):

    def __init__(self,
                 input_shape: Tuple[int],
                 Phi_sizes: List[int],
                 F_sizes: List[int],
                 output_layer: tf.keras.layers.Layer,
                 backbone: Literal["cnn", "fc"] = "cnn",
                 batch_norm: bool = False,
                 Phi_dropout: Optional[float] = None,
                 F_dropout: Optional[float] = None,
                 activation: Optional[Callable] = tf.nn.relu,
                 preprocess: Optional[tf.keras.layers.Layer] = None):

        self.Phi_sizes, self.F_sizes = Phi_sizes, F_sizes
        self.Phi_dropout = Phi_dropout
        self.F_dropout = F_dropout
        self.activation = activation

        input = tf.keras.layers.Input(shape=input_shape, ragged=True)
        angular = input[:, :, 6:]
        energy = input[:, :, :6].to_tensor()

        row_lengths = angular.row_lengths()
        mask = tf.sequence_mask(row_lengths)
        hidden = angular.to_tensor()

        if preprocess is not None:
            hidden = preprocess(hidden)

        if batch_norm:
            hidden = tf.keras.layers.BatchNormalization()(hidden)

        if backbone == "cnn":
            hidden = self.cnn_Phi(hidden)
        elif backbone == "fc":
            hidden = self.fc_Phi(hidden)
        else:
            raise ValueError(f"backbone must be either 'cnn' or 'fc', not {backbone}")

        hidden = hidden * tf.expand_dims(tf.cast(mask, tf.float32), -1)
        hidden = EinsumLayer('BPC,BPD->BCD')((hidden, energy))
        hidden = tf.reshape(hidden, (-1, hidden.shape[-1] * hidden.shape[-2]))
        hidden = self.fc_F(hidden)
        output = output_layer(hidden)

        super().__init__(inputs=input, outputs=output)

    def cnn_Phi(self, inputs):
        hidden = inputs
        for size in self.Phi_sizes:
            hidden = tf.keras.layers.Conv1D(size, 1)(hidden)
            hidden = tf.keras.layers.BatchNormalization()(hidden)
            hidden = tf.keras.layers.Activation(self.activation)(hidden)
            if self.Phi_dropout is not None:
                hidden = tf.keras.layers.Dropout(self.Phi_dropout)(hidden)
        return hidden

    def fc_Phi(self, inputs):
        hidden = inputs
        for size in self.Phi_sizes:
            hidden = tf.keras.layers.Dense(size)(hidden)
            hidden = tf.keras.layers.Activation(self.activation)(hidden)
            if self.Phi_dropout is not None:
                hidden = tf.keras.layers.Dropout(self.Phi_dropout)(hidden)
        return hidden

    def fc_F(self, inputs):
        hidden = inputs
        for size in self.F_sizes:
            hidden = tf.keras.layers.Dense(size)(hidden)
            hidden = tf.keras.layers.Activation(self.activation)(hidden)
            if self.F_dropout is not None:
                hidden = tf.keras.layers.Dropout(self.F_dropout)(hidden)
        return hidden
