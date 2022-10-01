# from __future__ import annotations
import tensorflow as tf
from typing import Callable


class BasicFCModel(tf.keras.Model):
    def __init__(self, 
                 hidden_layers: list[int], 
                 input_size: tuple[int, int] | int,
                 output_layer: tf.keras.layers.Layer,
                 activation: Callable,
                 loss: tf.keras.losses.Loss,
                 metrics=list[tf.keras.metrics.Metric],
                 optimizer=tf.optimizers.Optimizer,
                 rnn_dim: int | None = None,
                 dropout: float | None = None,
                 preprocess: tf.keras.layers.Layer | None = None) -> None:
        
        self._activation = activation
            
        
        dropout_layer = tf.keras.layers.Dropout(dropout) if dropout is not None else None
        if isinstance(input_size, tuple) and len(input_size) == 2:
            inputs0 = tf.keras.layers.Input(shape=(input_size[0], ))
            inputs1 = tf.keras.layers.Input(shape=(None, input_size[1]), ragged=True)
            
            hidden = self._hidden_layers(preprocess(inputs0) if preprocess is not None else inputs0, hidden_layers, dropout_layer)
            densed = tf.keras.layers.Dense(rnn_dim, activation = self._activation)(hidden)
            
            rnned = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(rnn_dim), merge_mode='sum')(inputs1)
            
            rnned = tf.keras.layers.BatchNormalization()(rnned)
            
            hidden = tf.keras.layers.Concatenate()([densed, rnned])
            inputs = (inputs0, inputs1)
            
        elif isinstance(input_size, int):
            inputs = tf.keras.layers.Input(shape=(input_size, ))
            hidden = preprocess(inputs) if preprocess is not None else inputs
        else:
            raise ValueError("Input size must be len two tuple or int.")

        output = output_layer(hidden)
            
        super().__init__(inputs=inputs, outputs=output)
        
        self.compile(
            optimizer=optimizer,
            loss=loss,
            weighted_metrics=metrics,
        )

    def _hidden_layers(self, inputs, layers:list[int], dropout: tf.keras.layers.Dropout | None = None) -> tf.Tensor:
        hidden =  tf.keras.layers.Flatten()(inputs)
        for hidden_layer in layers:
            hidden = tf.keras.layers.Dense(hidden_layer, activation=self._activation)(hidden)
            if dropout is not None:
                hidden = dropout(hidden)
        return hidden
    

    
    
    
