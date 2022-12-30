# from __future__ import annotations
import tensorflow as tf
from typing import Callable, Union, Tuple

class BasicFCModel(tf.keras.Model):
    def __init__(self, 
                 hidden_layer_size: int,
                 num_layers: int,
                 input_size: Union[Tuple[int, int], int],
                 output_layer: tf.keras.layers.Layer,
                 activation: Callable,
                 rnn_dim: Union[int, None] = None,
                 dropout: Union[float, None] = None,
                 preprocess: Union[tf.keras.layers.Layer, None] = None) -> None:
        
        self._activation = activation
        
        if isinstance(input_size, tuple) and len(input_size) == 2:
            inputs0 = tf.keras.layers.Input(shape=(input_size[0], ))
            inputs1 = tf.keras.layers.Input(shape=(None, input_size[1]), ragged=True)
            
            densed = self._hidden_layers(preprocess(
                inputs0) if preprocess is not None else inputs0, hidden_layer_size, num_layers, dropout)
            
            rnned = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(rnn_dim), merge_mode='sum')(inputs1)
            
            hidden = tf.keras.layers.Concatenate()([densed, rnned])
            
            inputs = (inputs0, inputs1)
            
        elif isinstance(input_size, int):
            inputs = tf.keras.layers.Input(shape=(input_size, ))
            preprocessed = preprocess(inputs) if preprocess is not None else inputs
            # preprocessed = tf.keras.layers.LayerNormalization()(preprocessed)
            hidden = self._hidden_layers(preprocessed, hidden_layer_size, num_layers, dropout)
        else:
            raise ValueError("Input size must be len two tuple or int.")

        output = output_layer(hidden)
            
        super().__init__(inputs=inputs, outputs=output)
        

    def _hidden_layers(self, inputs, layer_size:int, num_layers:int,  dropout: Union[float, None] = None) -> tf.Tensor:
        hidden = inputs #tf.keras.layers.Flatten()(inputs)
        for _ in range(num_layers):
            hidden = tf.keras.layers.Dense(layer_size, activation=self._activation)(hidden)
            if dropout is not None:
                hidden = tf.keras.layers.Dropout(dropout)(hidden)
        return hidden
    
    
    
    
