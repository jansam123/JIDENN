from __future__ import annotations
import tensorflow as tf
from typing import Callable, List


class BasicFCModel(tf.keras.Model):
    def __init__(self, 
                 hidden_layers: List[int], 
                 input_layer: tf.keras.layers.Layer,
                 output_layer: tf.keras.layers.Layer,
                 activation: Callable,
                 loss: tf.keras.losses.Loss,
                 metrics=List[tf.keras.metrics.Metric],
                 optimizer=tf.optimizers.Optimizer,
                 preprocess: tf.keras.layers.Layer | None = None) -> None:
        
        self._activation = activation
        inputs = input_layer
        hidden = self._hidden_layers(preprocess(inputs) if preprocess else inputs, hidden_layers)
        output = output_layer(hidden)
            
        super().__init__(inputs=inputs, outputs=output)
        
        self.compile(
            optimizer=optimizer,
            loss=loss,
            weighted_metrics=metrics,
        )

    def _hidden_layers(self, inputs: tf.Tensor, layers:list[int]) -> tf.Tensor:
        hidden =  tf.keras.layers.Flatten()(inputs)
        for hidden_layer in layers:
            hidden = tf.keras.layers.Dense(hidden_layer, activation=self._activation)(hidden)
        return hidden
    
    
    
