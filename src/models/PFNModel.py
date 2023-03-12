import tensorflow as tf
from typing import Optional, Union, List, Literal, Tuple


class PFNModel(tf.keras.Model):

    def __init__(self,
                 input_shape: Tuple[int],
                 Phi_sizes: List[int],
                 F_sizes: List[int],
                 output_layer: tf.keras.layers.Layer,
                 backbone: Literal["cnn", "fc"] = "cnn",
                 batch_norm: bool = False,
                 preprocess: Optional[tf.keras.layers.Layer] = None):
        self.Phi_sizes, self.F_sizes = Phi_sizes, F_sizes

        input = tf.keras.layers.Input(shape=input_shape, ragged=True)
        row_lengths = input.row_lengths()
        mask = tf.sequence_mask(row_lengths)
        hidden = input.to_tensor()
        
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
        hidden = tf.math.reduce_sum(hidden, axis=1)
        hidden = self.fc_F(hidden)
        output = output_layer(hidden)

        super().__init__(inputs=input, outputs=output)

    def cnn_Phi(self, inputs):
        hidden = inputs
        for size in self.F_sizes:
            hidden = tf.keras.layers.Conv1D(size, 1)(hidden)
            hidden = tf.keras.layers.BatchNormalization()(hidden)
            hidden = tf.keras.layers.ReLU()(hidden)
        return hidden

    def fc_Phi(self, inputs):
        hidden = inputs
        for size in self.F_sizes:
            hidden = tf.keras.layers.Dense(size)(hidden)
            hidden = tf.keras.layers.ReLU()(hidden)
        return hidden

    def fc_F(self, inputs):
        hidden = inputs
        for size in self.F_sizes:
            hidden = tf.keras.layers.Dense(size)(hidden)
            hidden = tf.keras.layers.ReLU()(hidden)
        return hidden
