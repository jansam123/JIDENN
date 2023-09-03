"""
Module for initializing normalization layers and adapting them to the dataset,
i.e. calculating the mean and std of the dataset.
"""

import tensorflow as tf
import logging
from typing import Optional, Tuple, Union, Literal


def get_normalization(dataset: tf.data.Dataset,
                      log: logging.Logger,
                      input_shape,
                      adapt: bool = True,
                      normalization_steps: Optional[int] = None,) -> Union[tf.keras.layers.Normalization, Tuple[tf.keras.layers.Normalization, tf.keras.layers.Normalization]]:
    """Function returning normalization layer(s) adapted to the dataset if requested.
    Unadapted normalization layer(s) are asumed to have weights loaded from checkpoint.

    Args:
        dataset (tf.data.Dataset): Dataset to adapt the normalization layer to.
        log (logging.Logger): Logger.
        input_shape (Tuple[None, int]): The shape of the input.
        adapt (bool, optional): Whether to adapt the normalization layer to the dataset. Defaults to True.
        normalization_steps (int, optional): Number of batches to use for adaptation. Defaults to None.

    Returns:
        Union[tf.keras.layers.Normalization, Tuple[tf.keras.layers.Normalization, tf.keras.layers.Normalization]]: Return `None` if the model is `bdt` or unknown model,
            one layer if there is no interaction, two layers as a `tuple` if there is interaction.
    """
    ragged = isinstance(input_shape, tuple) and (None in input_shape or None in input_shape[0])
    gnn = isinstance(input_shape, tuple) and isinstance(input_shape[0], tuple) and (None not in input_shape[0])
    interaction = isinstance(input_shape, tuple) and isinstance(input_shape[0], tuple) and len(input_shape[1]) == 3

    if not adapt:
        log.warning("Normalization not adapting. Loading weights expected.")
        if not interaction:
            return tf.keras.layers.Normalization(axis=-1)
        else:
            return tf.keras.layers.Normalization(axis=-1), tf.keras.layers.Normalization(axis=-1)

    normalizer = tf.keras.layers.Normalization(axis=-1)

    log.info("Getting std and mean of the dataset...")
    log.info(f"Subsample size (num of batches): {normalization_steps}")

    if not ragged and not gnn:
        def picker(*x):
            return x[0]
    elif not ragged and gnn:
        def picker(*x):
            return x[0][0]
    elif isinstance(input_shape, tuple) and isinstance(input_shape[0], tuple):
        def picker(*x):
            return x[0][0].to_tensor()
    else:
        def picker(*x):
            return x[0].to_tensor()

    try:
        normalizer.adapt(dataset.map(picker), steps=normalization_steps)
    except Exception as e:
        log.error(f"Normalization failed: {e}")
        normalizer = None

    if not interaction and not gnn:
        log.info("No interaction normalization")
        return normalizer
    elif not interaction and gnn:
        normalizer_fts = tf.keras.layers.Normalization(axis=-1)

        def picker2(*x):
            return x[0][1]
        normalizer_fts.adapt(dataset.map(picker2), steps=normalization_steps)
        return normalizer, normalizer_fts

    log.info("Getting std and mean of the **interaction** dataset...")

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
        normalizer_interaction.adapt(dataset.map(
            picker_interaction), steps=normalization_steps)
    except Exception as e:
        log.error(f"Interaction normalization failed: {e}")
        normalizer_interaction = None
    return normalizer, normalizer_interaction
