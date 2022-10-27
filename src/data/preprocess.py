import tensorflow as tf
from typing import Callable
from src.config import config_subclasses as cfg


def pipe(name: str,
         args_dataset: cfg.Dataset,
         take: int | None) -> Callable[[tf.data.Dataset], tf.data.Dataset]:
    def _pipe(dataset: tf.data.Dataset) -> tf.data.Dataset:

        if args_dataset.shuffle_buffer is not None and name == 'train':
            dataset = dataset.shuffle(buffer_size=args_dataset.shuffle_buffer)
            
        if take is not None:
            dataset = dataset.take(take)
            dataset = dataset.apply(tf.data.experimental.assert_cardinality(take)) if name == 'train' else dataset

        # dataset = dataset.snapshot(f"rc.cache.{name}")
        dataset = dataset.cache()
        dataset = dataset.batch(args_dataset.batch_size)
        # dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args_dataset.batch_size))
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset
    return _pipe
