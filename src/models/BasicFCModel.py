# from __future__ import annotations
import tensorflow as tf
from typing import Callable


class BasicFCModel(tf.keras.Model):
    def __init__(self, 
                 hidden_layers: list[int], 
                 input_layer: tf.keras.layers.Layer,
                 output_layer: tf.keras.layers.Layer,
                 activation: Callable,
                 loss: tf.keras.losses.Loss,
                 metrics=list[tf.keras.metrics.Metric],
                 optimizer=tf.optimizers.Optimizer,
                 dropout: float | None = None,
                 preprocess: tf.keras.layers.Layer | None = None) -> None:
        
        self._activation = activation
        inputs = input_layer
        
        embed = tf.keras.layers.Embedding(input_dim=100, output_dim=16)(preprocess(inputs) if preprocess else inputs)
        dropout_layer = tf.keras.layers.Dropout(dropout) if dropout is not None else None
        hidden = self._hidden_layers(embed, hidden_layers, dropout_layer)
        output = output_layer(hidden)
            
        super().__init__(inputs=inputs, outputs=output)
        
        self.compile(
            optimizer=optimizer,
            loss=loss,
            weighted_metrics=metrics,
        )

    def _hidden_layers(self, inputs: tf.Tensor, layers:list[int], dropout: tf.keras.layers.Dropout | None = None) -> tf.Tensor:
        hidden =  tf.keras.layers.Flatten()(inputs)
        for hidden_layer in layers:
            hidden = tf.keras.layers.Dense(hidden_layer, activation=self._activation)(hidden)
            if dropout is not None:
                hidden = dropout(hidden)
        return hidden
    
    
    
