import tensorflow as tf

from src.config import config_subclasses as cfg
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


def create(args: cfg.Params, args_model: cfg.Transformer, args_data: cfg.Data, preprocess: tf.keras.layers.Layer | None = None) -> TransformerModel:

    activation = tf.nn.relu

    if args_data.num_labels == 2:
        output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=args.label_smoothing)
        metrics = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()]
    else:
        output = tf.keras.layers.Dense(args_data.num_labels, activation=tf.nn.softmax)
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing)
        metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC()]

    # tf.optimizers.schedules.CosineDecay(args.learning_rate, args.decay_steps)
    l_r = LinearWarmup(args_model.warmup_steps, lambda x: args.learning_rate)
    optimizer = tf.optimizers.Adam(learning_rate=l_r)

    model = TransformerModel(
        input_shape=(None, len(args_data.variables.perJetTuple)),
        output_layer=output,
        #
        embedding_dim=args_model.embed_dim,
        transformer_layers=args_model.transformer_layers,
        transformer_expansion=args_model.transformer_expansion,
        transformer_heads=args_model.transformer_heads,
        transformer_dropout=args_model.transformer_dropout,
        last_hidden_layer=args_model.last_hidden_layer,
        #
        preprocess=preprocess,
        activation=activation
    )
    model.compile(optimizer=optimizer,
                  loss=loss,
                  weighted_metrics=metrics,)
    return model
