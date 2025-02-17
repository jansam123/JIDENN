"""
Module defining functions to be used as decorators for functions that build models.
They are used to choose the strategy for multi-gpu training.
"""

import tensorflow as tf
import keras
from typing import Any, Callable


def choose_strategy(model_builder: Callable[...,  keras.Model], num_gpus: int) -> Callable[...,  keras.Model]:
    """Decorator that chooses strategy based on the number of available GPUs.
    If there is only one GPU, no strategy is used.
    If there are more than one GPUs, `tf.distribute.MirroredStrategy` is used.
    
    Example:
    ```python
    import tensorflow as tf
    from functools import partial
    
    num_gpus = len(gpus = tf.config.list_physical_devices("GPU"))
    gpu_strategy = partial(choose_strategy, num_gpus=num_gpus)
    
    @gpu_strategy
    def model_builder(...) -> keras.Model:
        ...
        return model
    
    model = model_builder(...)
    ```

    Args:
        model_builder (Callable[...,  keras.Model]): Function that builds the model.
        num_gpus (int): Number of available GPUs.

    Returns:
        Callable[...,  keras.Model]: Function that builds the model in the scope of chosen strategy.
    """
    def _choose_strategy_wrapper(*args: Any, **kwargs: Any) -> keras.Model:
        if num_gpus < 2:
            model = model_builder(*args, **kwargs)
        else:
            # mirrored_strategy = tf.distribute.MirroredStrategy(devices=[f"/gpu:{i}" for i in range(num_gpus)], cross_device_ops=tf.distribute.ReductionToOneDevice())
            mirrored_strategy = tf.distribute.MirroredStrategy()
            assert mirrored_strategy.num_replicas_in_sync == num_gpus, "Not all GPUs are recognized"
            with mirrored_strategy.scope():
                model = model_builder(*args, **kwargs)
        return model
    return _choose_strategy_wrapper
