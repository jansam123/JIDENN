import tensorflow as tf
from typing import Callable
from src.config import config_subclasses as cfg        


def pipe(datasets: list[tf.data.Dataset],
                             dataset_weights: list[float],
                             args_dataset: cfg.Dataset,
                             label_mapping: Callable,
                             take: int | None) -> tf.data.Dataset:
    
    assert len(datasets) == len(dataset_weights), "Number of datasets and weights must be equal."

        
    prep_datasets = [ds.take(int(take/len(datasets))) for ds in datasets] if take is not None else datasets
    dataset = tf.data.Dataset.sample_from_datasets(prep_datasets, weights=dataset_weights)
    
    if label_mapping is not None:
        dataset = dataset.map(lambda x,y,z: (x, label_mapping(y),z))
    
        
    if take is not None:
        dataset = dataset.take(take)
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(take)) 

        
    dataset = dataset.shuffle(buffer_size=args_dataset.shuffle_buffer) if args_dataset.shuffle_buffer is not None else dataset
    dataset = dataset.batch(args_dataset.batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset