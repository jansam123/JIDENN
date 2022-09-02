import tensorflow as tf
from ..config.ArgumentParser import ArgumentParser
from typing import Optional, Callable


class BasicFCModel(tf.keras.Model):
    def __init__(self, args: ArgumentParser, input_size: int, output_layer: int, preprocess: Optional[Callable[[tf.Tensor], tf.Tensor]]=None) -> None:
        
        inputs = tf.keras.layers.Input(shape=(input_size,))
        if preprocess is not None:
            hidden = preprocess(inputs)
        else:
            hidden = inputs
        hidden = self._hidden_layers(hidden, args.hidden_layers)
        if output_layer == 2:
            output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(hidden)
            loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=args.label_smoothing)
            metrics = [tf.keras.metrics.BinaryAccuracy()]
                                                
        else:
            output = tf.keras.layers.Dense(output_layer, activation=tf.nn.softmax)(hidden)
            loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing)
            metrics = [tf.keras.metrics.CategoricalAccuracy()]
            
        super().__init__(inputs=inputs, outputs=output)
        
        self.compile(
            optimizer=tf.optimizers.Adam(),
            loss=loss,
            weighted_metrics=metrics,
        )

    def _hidden_layers(self, inputs: tf.Tensor, layers:list[int]) -> tf.Tensor:
        hidden =  tf.keras.layers.Flatten()(inputs)
        for hidden_layer in layers:
            hidden = tf.keras.layers.Dense(hidden_layer, activation=tf.nn.relu)(hidden)
        return hidden
    