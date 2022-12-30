import tensorflow as tf
import tensorflow_addons as tfa
from typing import Union

from .optimizers import LinearWarmup
from src.config import config_subclasses as cfg
from .DeParTModel import DeParTModel


def create(args: cfg.Params, args_model: cfg.DeParT, args_data: cfg.Data, preprocess: Union[tf.keras.layers.Layer, None] = None) -> DeParTModel:

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

    # optimizer = tfa.optimizers.RectifiedAdam(learning_rate=l_r, weight_decay=args.weight_decay if args.weight_decay is not None else 0, beta_1=args.beta_1, beta_2=args.beta_2, epsilon=args.epsilon)
    # optimizer = tfa.optimizers.Lookahead(optimizer, sync_period=args_model.lookahead_sync_period, slow_step_size=args_model.lookahead_slow_step_size)
    # if args.weight_decay is not None and args.weight_decay > 0:
    #     optimizer = tfa.optimizers.AdamW(learning_rate=l_r, weight_decay=args.weight_decay,
    #                                      beta_1=args.beta_1, beta_2=args.beta_2, epsilon=args.epsilon if args.epsilon is not None else 1e-7)
    # else:
    #     optimizer = tf.optimizers.Adam(learning_rate=l_r, beta_1=args.beta_1, beta_2=args.beta_2, epsilon=args.epsilon if args.epsilon is not None else 1e-7)
    optimizer = tfa.optimizers.LAMB(learning_rate=l_r,
                                    weight_decay=args.weight_decay if args.weight_decay is not None else 0,
                                    beta_1=args.beta_1 if args.beta_1 is not None else 0.9,
                                    beta_2=args.beta_2 if args.beta_2 is not None else 0.999,
                                    epsilon=args.epsilon if args.epsilon is not None else 1e-6,
                                    clipnorm=args.clip_norm)

    model = DeParTModel(
        input_shape=(None, args_data.input_size),
        output_layer=output,
        #
        embedding_dim=args_model.embed_dim,
        num_embeding_layers=args_model.num_embed_layers,
        layers= args_model.layers,
        class_layers=args_model.class_layers,
        expansion=args_model.expansion,
        heads=args_model.heads,
        dropout=args_model.dropout,
        interaction=args_model.interaction,
        layer_scale_init_value=args_model.layer_scale_init_value,
        stochastic_depth_drop_rate=args_model.stochastic_depth_drop_rate,
        #
        preprocess=preprocess,
        activation=activation
    )
    model.compile(optimizer=optimizer,
                  loss=loss,
                  weighted_metrics=metrics,)
    return model
