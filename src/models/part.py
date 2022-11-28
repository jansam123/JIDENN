import tensorflow as tf
import tensorflow_addons as tfa
from typing import Union

from src.config import config_subclasses as cfg
from src.models.ParTModel import ParTModel


class LinearWarmup(tf.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps, following_schedule):
        self._warmup_steps = warmup_steps
        self._warmup = tf.optimizers.schedules.PolynomialDecay(0., warmup_steps, following_schedule(0))
        self._following = following_schedule

    def get_config(self):
        config = {
            'warmup_steps': self._warmup_steps,
            'following_schedule': self._following
        }
        return config

    def __call__(self, step):
        return tf.cond(step < self._warmup_steps,
                       lambda: self._warmup(step),
                       lambda: self._following(step - self._warmup_steps))


def create(args: cfg.Params, args_model: cfg.ParT, args_data: cfg.Data, preprocess: Union[tf.keras.layers.Layer, None] = None) -> ParTModel:

    activations = {'relu': tf.nn.relu, 'elu': tf.nn.elu, 'gelu': tf.nn.gelu, 'silu': tf.nn.silu}
    activation = activations[args.activation]

    if args_data.num_labels == 2:
        output = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=args.label_smoothing)
        metrics = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()]
    else:
        output = tf.keras.layers.Dense(args_data.num_labels, activation=tf.nn.softmax)
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing)
        metrics = [tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.AUC()]

    scheduler = tf.keras.optimizers.schedules.CosineDecay(
        args.learning_rate, args.decay_steps) if args.decay_steps is not None else lambda x: args.learning_rate
    l_r = LinearWarmup(args_model.warmup_steps, scheduler)

    if args.weight_decay is not None and args.weight_decay > 0:
        optimizer = tfa.optimizers.AdamW(learning_rate=l_r, weight_decay=args.weight_decay)
    else:
        optimizer = tf.optimizers.Adam(learning_rate=l_r)

    model = ParTModel(
        input_shape=(None, args_data.input_size),
        output_layer=output,
        #
        embedding_dim=args_model.embed_dim,
        num_embeding_layers=args_model.num_embed_layers,
        particle_block_layers=args_model.particle_block_layers,
        class_block_layers=args_model.class_block_layers,
        transformer_expansion=args_model.transformer_expansion,
        transformer_heads=args_model.transformer_heads,
        particle_block_dropout=args_model.particle_block_dropout,
        #
        preprocess=preprocess,
        activation=activation
    )
    model.compile(optimizer=optimizer,
                  loss=loss,
                  weighted_metrics=metrics,)
    return model
