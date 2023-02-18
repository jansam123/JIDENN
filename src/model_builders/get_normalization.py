import tensorflow as tf
import numpy as np
import logging
from typing import Optional, Tuple, Union


def get_normalization(model_name: str,
                      dataset: tf.data.Dataset,
                      log: logging.Logger,
                      adapt: bool = True,
                      normalization_steps: Optional[int] = None,
                      interaction: Optional[bool] = None,) -> Union[tf.keras.layers.Normalization, Tuple[tf.keras.layers.Normalization, tf.keras.layers.Normalization], None]:
    if model_name == 'bdt':
        log.info("No normalization for BDT")
        return None

    if not adapt:
        log.info("Normalization not adapting. Loading weights expected.")
        if not interaction:
            return tf.keras.layers.Normalization(axis=-1)
        else:
            return tf.keras.layers.Normalization(axis=-1), tf.keras.layers.Normalization(axis=-1)

    normalizer = tf.keras.layers.Normalization(axis=-1)

    log.info("Getting std and mean of the dataset...")
    log.info(f"Subsample size: {normalization_steps}")

    if model_name in ['basic_fc', 'highway']:
        def picker(*x): return x[0]

    elif model_name in ['transformer', 'part', 'depart']:
        if interaction:
            def picker(*x): return x[0][0].to_tensor()
        else:
            def picker(*x): return x[0].to_tensor()
    else:
        log.error(f"Unknown model {model_name}")
        return None

    try:
        normalizer.adapt(dataset.map(picker), steps=normalization_steps)
    except Exception as e:
        log.error(f"Normalization failed: {e}")
        normalizer = None

    if not interaction:
        log.info("No interaction normalization")
        return normalizer

    log.info("Getting std and mean of the interaction dataset...")

    def picker_interaction(*x):
        inputs = x[0][1].to_tensor()
        ones = tf.ones_like(inputs[:, :, 0])
        upper_tril_mask = tf.linalg.band_part(ones, 0, -1)
        diag_mask = tf.linalg.band_part(ones, 0, 0)
        upper_tril_mask = tf.cast(upper_tril_mask - diag_mask, tf.bool)
        flattened_upper_triag = tf.boolean_mask(inputs, upper_tril_mask)
        return flattened_upper_triag

    normalizer_interaction = tf.keras.layers.Normalization(axis=-1)
    try:
        normalizer_interaction.adapt(dataset.map(picker_interaction), steps=normalization_steps)
    except Exception as e:
        log.error(f"Interaction normalization failed: {e}")
        normalizer_interaction = None
    return normalizer, normalizer_interaction
