import tensorflow as tf
from typing import Callable
from src.config import config_subclasses as cfg        


def pipe(datasets: list[tf.data.Dataset],
                             dataset_weights: list[float],
                             args_dataset: cfg.Dataset,
                             label_mapping: Callable,
                             name: str) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    
    assert len(datasets) == len(dataset_weights), "Number of datasets and weights must be equal."
    
    dev_size = int(args_dataset.take*args_dataset.dev_size) if args_dataset.take is not None else None
    test_size = int(args_dataset.take*args_dataset.test_size) if args_dataset.take is not None else None
    
    
    # prep_datasets = [ds.take(int(args_dataset.take/len(datasets))) for ds in datasets] if args_dataset.take is not None else datasets
    dataset = tf.data.Dataset.sample_from_datasets(datasets, weights=dataset_weights)
        
    if label_mapping is not None:
        dataset = dataset.map(lambda x,y,z: (x, label_mapping(y),z))
    
        
    if args_dataset.take is not None:
        dataset = dataset.take(args_dataset.take)
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(args_dataset.take)) 

        
    dataset = dataset.snapshot(f"rc.cache.{name}")
    dataset = dataset.shuffle(buffer_size=args_dataset.shuffle_buffer) if args_dataset.shuffle_buffer is not None else dataset
    
    dev = dataset.take(dev_size).batch(args_dataset.batch_size)
    dataset = dataset.skip(dev_size)
    test = dataset.take(test_size).batch(args_dataset.batch_size)
    dataset = dataset.skip(test_size)
    dataset = dataset.batch(args_dataset.batch_size)
    # dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(args_dataset.batch_size))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dev = dev.prefetch(tf.data.AUTOTUNE)
    test = test.prefetch(tf.data.AUTOTUNE)
    
    return dataset, dev, test