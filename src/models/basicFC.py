import tensorflow as tf

from src.config import config_subclasses as cfg
from .BasicFCModel import BasicFCModel



def create(args: cfg.Params, args_model: cfg.BasicFC, args_data: cfg.Data, preprocess: tf.keras.layers.Layer | None  = None) -> BasicFCModel:
    activation = tf.nn.relu
    input0_size = len(args_data.variables.perJet)
    input0_size += len(args_data.variables.perEvent) if args_data.variables.perEvent is not None else 0
    inputs0 = tf.keras.layers.Input(shape=(input0_size, ))
    inputs1 = tf.keras.layers.Input(shape=(None, len(args_data.variables.perJetTuple)), ragged=True) 
    
    if args_data.num_labels == 2:
        output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=args.label_smoothing)
        metrics = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()]
                                            
    else:
        output = tf.keras.layers.Dense(args_data.num_labels, activation=tf.nn.softmax)
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing)
        metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC()]
    
    optimizer = tf.optimizers.Adam(learning_rate=args.learning_rate)


    model = BasicFCModel(
        hidden_layers=args_model.hidden_layers,
        dropout=args_model.dropout,
        input_layer=(inputs0, inputs1),
        output_layer=output,
        activation=activation,
        loss=loss,
        metrics=metrics,
        preprocess=preprocess,
        optimizer=optimizer)
    return model