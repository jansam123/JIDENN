import tensorflow as tf
from typing import Optional

from src.config.ArgumentParser import ArgumentParser
from src.models.TransformerModel import TransformerModel


class LinearWarmup(tf.optimizers.schedules.LearningRateSchedule):
            def __init__(self, warmup_steps, following_schedule):
                self._warmup_steps = warmup_steps
                self._warmup = tf.optimizers.schedules.PolynomialDecay(0., warmup_steps, following_schedule(0))
                self._following = following_schedule

            def __call__(self, step):
                return tf.cond(step < self._warmup_steps,
                            lambda: self._warmup(step),
                            lambda: self._following(step - self._warmup_steps))

def create(args: ArgumentParser, preprocess: Optional[tf.keras.layers.Layer] = None) -> TransformerModel:
     
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
        
    l_r = LinearWarmup(args.warmup_steps, tf.optimizers.schedules.CosineDecay(args.learning_rate, args.decay_steps))
    optimizer = tf.optimizers.Adam(learning_rate=l_r)
    
        
    return TransformerModel(
        args=args,
        input_layer=inputs,
        output_layer=output,
        loss=loss,
        metrics=metrics,
        preprocess=preprocess,
        optimizer=optimizer,
        activation=activation
    )