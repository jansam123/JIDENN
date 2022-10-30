import tensorflow as tf

from src.config import config_subclasses as cfg
from .HighwayModel import HighwayModel


def create(args: cfg.Params, args_model: cfg.Highway, args_data: cfg.Data, preprocess: tf.keras.layers.Layer | None = None) -> HighwayModel:
    activations = {'relu': tf.nn.relu, 'elu': tf.nn.elu, 'gelu': tf.nn.gelu, 'silu': tf.nn.silu}
    activation = activations[args.activation]
    input_size = len(args_data.variables.perJet)
    input_size += len(args_data.variables.perEvent) if args_data.variables.perEvent is not None else 0

    if args_data.num_labels == 2:
        output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=args.label_smoothing, name='bce')
        metrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy'), tf.keras.metrics.AUC(name='auc')]

    else:
        output = tf.keras.layers.Dense(args_data.num_labels, activation=tf.nn.softmax)
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing)
        metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC()]

    l_r = tf.keras.optimizers.schedules.CosineDecay(args.learning_rate, args.decay_steps) if args.decay_steps is not None else args.learning_rate
    optimizer = tf.optimizers.Adam(learning_rate=l_r)
    # optimizer = tf.optimizers.SGD(learning_rate=args.learning_rate, momentum=0.1)

    model = HighwayModel(
        layer_size=args_model.layer_size,
        num_layers=args_model.num_layers,
        dropout=args_model.dropout,
        input_size=input_size,
        output_layer=output,
        activation=activation,
        preprocess=preprocess)

    model.compile(
        loss=loss,
        weighted_metrics=metrics,
        optimizer=optimizer)

    return model
