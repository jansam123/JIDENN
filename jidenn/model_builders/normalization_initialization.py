"""
Module for initializing normalization layers and adapting them to the dataset,
i.e. calculating the mean and std of the dataset.
"""

import tensorflow as tf
import numpy as np
import keras
import logging
from typing import Optional, Tuple, Union, Literal

class MinMaxScaler(keras.layers.Layer):
    def __init__(self, feature_range=(0.0, 1.0), axis=None, **kwargs):
        """
        feature_range: tuple (min, max) desired range of transformed data.
        axis: int or list/tuple of ints specifying the axis/axes that should NOT be reduced.
              For example, if inputs are of shape (N, N, C) and axis=-1 or axis=[-1],
              then the min and max are computed over axes 0 and 1 (per-channel scaling).
        """
        super(MinMaxScaler, self).__init__(**kwargs)
        self.feature_range = feature_range
        # Normalize axis to a tuple of non-negative ints (if provided)
        if axis is None:
            self.non_reduce_axes = None
        elif isinstance(axis, int):
            self.non_reduce_axes = (axis,)
        else:
            self.non_reduce_axes = tuple(axis)
        self.data_min = None
        self.data_max = None

    def adapt(self, dataset: tf.data.Dataset):
        """
        Compute the minimum and maximum values over the dataset.
        
        The dataset is expected to yield tensors (or batches of tensors).
        The reduction is performed over all axes not specified in self.non_reduce_axes.
        """
        # Grab one batch to determine the rank and compute reduction axes.
        sample = next(iter(dataset))
        # If the dataset yields batches, sample should have shape (batch_size, ...).
        data_rank = len(sample.shape)
        if self.non_reduce_axes is None:
            reduction_axes = list(range(data_rank))
        else:
            # Convert negative axes to positive.
            non_reduce = [a if a >= 0 else data_rank + a for a in self.non_reduce_axes]
            reduction_axes = [i for i in range(data_rank) if i not in non_reduce]

        global_min = None
        global_max = None

        for batch in dataset:
            # Ensure the batch is a tensor.
            batch = tf.convert_to_tensor(batch, dtype=tf.float32)
            batch_min = tf.reduce_min(batch, axis=reduction_axes, keepdims=True)
            batch_max = tf.reduce_max(batch, axis=reduction_axes, keepdims=True)
            if global_min is None:
                global_min = batch_min
                global_max = batch_max
            else:
                global_min = tf.minimum(global_min, batch_min)
                global_max = tf.maximum(global_max, batch_max)
        
        self.data_min = global_min
        self.data_max = global_max
    
    def call(self, inputs):
        """
        Scale the inputs using the min and max computed in adapt.
        """
        if self.data_min is None or self.data_max is None:
            raise ValueError("MinMaxScaler has not been adapted to data yet. Call 'adapt' first.")
            
        scale = self.feature_range[1] - self.feature_range[0]
        # Prevent division by zero if data_max equals data_min.
        data_range = tf.where(
            tf.equal(self.data_max - self.data_min, 0),
            tf.ones_like(self.data_max - self.data_min),
            self.data_max - self.data_min
        )
        scaled = (inputs - self.data_min) / data_range
        return self.feature_range[0] + scaled * scale

def get_normalization(dataset: tf.data.Dataset,
                      log: logging.Logger,
                      input_shape,
                      adapt: bool = True,
                      min_max_normalization: bool = False,
                      normalization_steps: Optional[int] = None,) -> Union[keras.layers.Layer, Tuple[keras.layers.Layer, keras.layers.Layer]]:
    """Function returning normalization layer(s) adapted to the dataset if requested.
    Unadapted normalization layer(s) are asumed to have weights loaded from checkpoint.

    Args:
        dataset (tf.data.Dataset): Dataset to adapt the normalization layer to.
        log (logging.Logger): Logger.
        input_shape (Tuple[None, int]): The shape of the input.
        adapt (bool, optional): Whether to adapt the normalization layer to the dataset. Defaults to True.
        normalization_steps (int, optional): Number of batches to use for adaptation. Defaults to None.

    Returns:
        Union[keras.layers.Normalization, Tuple[keras.layers.Normalization, keras.layers.Normalization]]: Return `None` if the model is `bdt` or unknown model,
            one layer if there is no interaction, two layers as a `tuple` if there is interaction.
    """
    np.set_printoptions(precision=4,threshold=100)
    zero_dim = not isinstance(input_shape, tuple)
    single_tuple = isinstance(input_shape, tuple) and not isinstance(input_shape[0], tuple)
    
    if not min_max_normalization:
        log.info("Getting std and mean of the dataset...")
    else:
        log.info("Getting min and max of the dataset...")
    log.info(f"Subsample size (num of batches): {normalization_steps}")

    if zero_dim or single_tuple:
        normalizer = keras.layers.Normalization(axis=-1) if not min_max_normalization else MinMaxScaler(axis=-1)
        if not adapt:
            log.warning("Normalization not adapting. Loading weights expected.")
            return normalizer
        picker = lambda *x: x[0] if zero_dim else x[0][0]
        normalizer.adapt(dataset.map(picker).take(normalization_steps))
        if not min_max_normalization:
            log.info(f'Calculated mean: {normalizer.mean.numpy().flatten()}')
            log.info(f'Calculated std: {normalizer.variance.numpy().flatten()}')
        else:
            log.info(f'Calculated min: {normalizer.data_min.numpy().flatten()}')
            log.info(f'Calculated max: {normalizer.data_max.numpy().flatten()}')
        log.info(f'Returning 1 normalizer.')
        return normalizer
        
    else:
        num_tuples = len(input_shape)
        normalizers = [keras.layers.Normalization(axis=-1) if not min_max_normalization else MinMaxScaler(axis=-1) for _ in range(num_tuples)]
        if not adapt:
            log.warning("Normalization not adapting. Loading weights expected.")
            return tuple(normalizers)
        for i, normalizer in enumerate(normalizers):
            picker = lambda *x: x[0][i] 
            normalizer.adapt(dataset.map(picker).take(normalization_steps))
            
            if not min_max_normalization:
                log.info(f'Calculated {i} mean: {normalizer.mean.numpy().flatten()}')
                log.info(f'Calculated {i} std: {normalizer.variance.numpy().flatten()}')
            else:
                log.info(f'Calculated {i} min: {normalizer.data_min.numpy().flatten()}')
                log.info(f'Calculated {i} max: {normalizer.data_max.numpy().flatten()}')
        log.info(f'Returning {len(normalizers)} normalizers.')
        return tuple(normalizers)
        
        

