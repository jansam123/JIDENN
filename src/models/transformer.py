import tensorflow as tf
import tensorflow_addons as tfa
from typing import Union

from src.config import config_subclasses as cfg
from .TransformerModel import TransformerModel 
from .optimizers import LinearWarmup


def create(args: cfg.Params, args_model: cfg.Transformer, args_data: cfg.Data, preprocess: Union[tf.keras.layers.Layer, None] = None) -> TransformerModel:

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

    optimizer = tfa.optimizers.LAMB(learning_rate=l_r, 
                                    weight_decay=args.weight_decay if args.weight_decay is not None else 0, 
                                    beta_1=args.beta_1 if args.beta_1 is not None else 0.9, 
                                    beta_2=args.beta_2 if args.beta_2 is not None else 0.999,
                                    epsilon=args.epsilon if args.epsilon is not None else 1e-6,
                                    clipnorm=args.clip_norm)
    
    # if args.weight_decay is not None and args.weight_decay > 0:
    #     optimizer = tfa.optimizers.AdamW(learning_rate=l_r, weight_decay=args.weight_decay, beta_1=args.beta_1, beta_2=args.beta_2)
    # else:
    #     optimizer = tf.optimizers.Adam(learning_rate=l_r, beta_1=args.beta_1, beta_2=args.beta_2)

    model = TransformerModel(
        input_shape=(None, args_data.input_size),
        output_layer=output,
        #
        embedding_dim=args_model.embed_dim,
        num_embeding_layers=args_model.num_embed_layers,
        transformer_layers=args_model.transformer_layers,
        transformer_expansion=args_model.transformer_expansion,
        transformer_heads=args_model.transformer_heads,
        transformer_dropout=args_model.transformer_dropout,
        #
        preprocess=preprocess,
        activation=activation
    )
    model.compile(optimizer=optimizer,
                  loss=loss,
                  weighted_metrics=metrics,)
    return model
