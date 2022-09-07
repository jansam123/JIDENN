import tensorflow as tf
from typing import Optional

from ..config.ArgumentParser import ArgumentParser
from .BasicFCModel import BasicFCModel



def create(args: ArgumentParser, preprocess: Optional[tf.keras.layers.Layer]  = None) -> BasicFCModel:
    activation = tf.nn.relu
    inputs = tf.keras.layers.Input(shape=(args.input_size))
    
    if args.num_labels == 2:
        output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=args.label_smoothing)
        metrics = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()]
                                            
    else:
        output = tf.keras.layers.Dense(args.num_labels, activation=tf.nn.softmax)
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing)
        metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC()]
    
    optimizer = tf.optimizers.Adam(learning_rate=args.learning_rate)
    
    return BasicFCModel(
        hidden_layers=args.hidden_layers,
        input_layer=inputs,
        output_layer=output,
        activation=activation,
        loss=loss,
        metrics=metrics,
        preprocess=preprocess,
        optimizer=optimizer)