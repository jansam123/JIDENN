import tensorflow as tf

from src.config import config_subclasses as cfg
print(cfg.Params)
from .BasicFCModel import BasicFCModel



def create(args: cfg.Params, args_model: cfg.BasicFC, args_data: cfg.Data, preprocess: tf.keras.layers.Layer | None  = None) -> BasicFCModel:
    activation = tf.nn.relu
    inputs = tf.keras.layers.Input(shape=(args_data.input_size))
    
    if args_data.num_labels == 2:
        output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=args.label_smoothing)
        metrics = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()]
                                            
    else:
        output = tf.keras.layers.Dense(args_data.num_labels, activation=tf.nn.softmax)
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing)
        metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC()]
    
    optimizer = tf.optimizers.Adam(learning_rate=args.learning_rate)

    return BasicFCModel(
        hidden_layers=args_model.hidden_layers,
        dropout=args_model.dropout,
        input_layer=inputs,
        output_layer=output,
        activation=activation,
        loss=loss,
        metrics=metrics,
        preprocess=preprocess,
        optimizer=optimizer)