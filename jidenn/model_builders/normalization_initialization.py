"""
Module for initializing normalization layers and adapting them to the dataset,
i.e. calculating the mean and std of the dataset.
"""

import tensorflow as tf
import logging
from typing import Optional, Tuple, Union, Literal


def get_normalization(dataset: tf.data.Dataset,
                      log: logging.Logger,
                      adapt: bool = True,
                      ragged: bool = False,
                      normalization_steps: Optional[int] = None,
                      interaction: Optional[bool] = None,) -> Union[tf.keras.layers.Normalization, Tuple[tf.keras.layers.Normalization, tf.keras.layers.Normalization]]:
    """Function returning normalization layer(s) adapted to the dataset if requested.
    Unadapted normalization layer(s) are asumed to have weights loaded from checkpoint.

    Args:
        dataset (tf.data.Dataset): Dataset to adapt the normalization layer to.
        log (logging.Logger): Logger.
        adapt (bool, optional): Whether to adapt the normalization layer to the dataset. Defaults to True.
        ragged (bool, optional): Whether the dataset samples are ragged. Defaults to False.
        normalization_steps (int, optional): Number of batches to use for adaptation. Defaults to None.
        interaction (bool, optional): Whether to adapt the normalization layer to the interaction variables. Defaults to None.

    Returns:
        Union[tf.keras.layers.Normalization, Tuple[tf.keras.layers.Normalization, tf.keras.layers.Normalization]]: Return `None` if the model is `bdt` or unknown model,
            one layer if there is no interaction, two layers as a `tuple` if there is interaction.
    """

    if not adapt:
        log.info("Normalization not adapting. Loading weights expected.")
        if not interaction:
            return tf.keras.layers.Normalization(axis=-1)
        else:
            return tf.keras.layers.Normalization(axis=-1), tf.keras.layers.Normalization(axis=-1)

    normalizer = tf.keras.layers.Normalization(axis=-1)

    log.info("Getting std and mean of the dataset...")
    log.info(f"Subsample size (num of batches): {normalization_steps}")
    
    if not ragged:
        def picker(*x): return x[0]
    elif interaction:
        def picker(*x): return x[0][0].to_tensor()
    else:
        def picker(*x): return x[0].to_tensor()

    try:
        normalizer.adapt(dataset.map(picker), steps=normalization_steps)
    except Exception as e:
        log.error(f"Normalization failed: {e}")
        normalizer = None

    if not interaction:
        log.info("No interaction normalization")
        return normalizer
    
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
        normalizer_interaction.adapt(dataset.map(picker_interaction), steps=normalization_steps)
    except Exception as e:
        log.error(f"Interaction normalization failed: {e}")
        normalizer_interaction = None
    return normalizer, normalizer_interaction
