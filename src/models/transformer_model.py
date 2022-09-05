import tensorflow as tf
from src.config.ArgumentParser import ArgumentParser
from src.models.TransformerModel import TransformerModel


def create(args: ArgumentParser, preprocess: tf.keras.layers.Layer | None = None) -> TransformerModel:
    inputs = tf.keras.layers.Input(shape=(args.input_size))
    
    if args.num_labels == 2:
        output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=args.label_smoothing)
        metrics = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()]
    else:
        output = tf.keras.layers.Dense(args.num_labels, activation=tf.nn.softmax)
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing)
        metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC()]
        
        
    return TransformerModel(
        args=args,
        input_layer=inputs,
        output_layer=output,
        loss=loss,
        metrics=metrics,
        preprocess=preprocess,
    )