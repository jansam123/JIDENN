import tensorflow as tf
from typing import Optional, List, Literal, Tuple, Callable


class PFNModel(tf.keras.Model):

    def __init__(self,
                 input_shape: Tuple[int],
                 Phi_sizes: List[int],
                 F_sizes: List[int],
                 output_layer: tf.keras.layers.Layer,
                 Phi_backbone: Literal["cnn", "fc"] = "cnn",
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
        hidden = inputs
        for size in self.Phi_sizes:
            hidden = tf.keras.layers.Conv1D(size, 1)(hidden)
            hidden = tf.keras.layers.BatchNormalization()(hidden)
            hidden = tf.keras.layers.Activation(self.activation)(hidden)
            if self.Phi_dropout is not None:
                hidden = tf.keras.layers.Dropout(self.Phi_dropout)(hidden)
        return hidden

    def fc_Phi(self, inputs: tf.Tensor) -> tf.Tensor:
        hidden = inputs
        for size in self.Phi_sizes:
            hidden = tf.keras.layers.Dense(size)(hidden)
            hidden = tf.keras.layers.Activation(self.activation)(hidden)
            if self.Phi_dropout is not None:
                hidden = tf.keras.layers.Dropout(self.Phi_dropout)(hidden)
        return hidden

    def fc_F(self, inputs: tf.Tensor) -> tf.Tensor:
        hidden = inputs
        for size in self.F_sizes:
            hidden = tf.keras.layers.Dense(size)(hidden)
            hidden = tf.keras.layers.Activation(self.activation)(hidden)
            if self.F_dropout is not None:
                hidden = tf.keras.layers.Dropout(self.F_dropout)(hidden)
        return hidden
