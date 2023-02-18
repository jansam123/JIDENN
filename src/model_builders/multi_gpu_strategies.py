import tensorflow as tf
from typing import Any, Callable


def choose_strategy(model_builder: Callable[...,  tf.keras.Model], num_gpus: int) -> Callable[...,  tf.keras.Model]:
    def _choose_strategy_wrapper(*args: Any, **kwargs: Any) -> tf.keras.Model:
        if num_gpus < 2:
            model = model_builder(*args, **kwargs)
        else:
            mirrored_strategy = tf.distribute.MirroredStrategy()
            with mirrored_strategy.scope():
                model = model_builder(*args, **kwargs)
        return model
    return _choose_strategy_wrapper
