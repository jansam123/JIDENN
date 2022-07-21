import tensorflow as tf
import argparse


class BasicFCModel(tf.keras.Model):
    def __init__(self, args: argparse.Namespace, input_size: int, output_layer: int) -> None:
        
        inputs = tf.keras.layers.Input(shape=(input_size,))
        hidden = self._hidden_layers(inputs, args.hidden_layers)
        output = tf.keras.layers.Dense(output_layer, activation=tf.nn.softmax)(hidden)

        super().__init__(inputs=inputs, outputs=output)
        
        self.compile(
            optimizer=tf.optimizers.Adam(),
            loss=tf.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
            metrics=[tf.metrics.CategoricalAccuracy("accuracy")],
        )

    def _hidden_layers(self, inputs: tf.Tensor, layers:list[int]) -> tf.Tensor:
        hidden =  tf.keras.layers.Flatten()(inputs)
        for hidden_layer in layers:
            hidden = tf.keras.layers.Dense(hidden_layer, activation=tf.nn.relu)(hidden)
        return hidden