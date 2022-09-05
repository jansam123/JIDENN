import tensorflow as tf
from typing import Optional, Callable



def pipe(datasets: list[tf.data.Dataset],
                             dataset_weights: list[float],
                             batch_size:int, 
                             take:Optional[int]=None, 
                             shuffle_buffer:Optional[int]=None,
                             num_labels: Optional[int]=None,
                             label_mapping: Optional[Callable]=None) -> tf.data.Dataset:
    
    assert len(datasets) == len(dataset_weights), "Number of datasets and weights must be equal."

        
    prep_datasets = [ds.take(int(take/len(datasets))) for ds in datasets] if take is not None else datasets
    dataset = tf.data.Dataset.sample_from_datasets(prep_datasets, weights=dataset_weights)
    
    if label_mapping is not None:
        dataset = dataset.map(lambda x,y,z: (x, label_mapping(y),z))
    
    if num_labels is not None:
        def one_hot_dataset(data, label, weight):
            label = tf.one_hot(tf.cast(label, tf.int32), num_labels)
            return data, label, weight
        dataset = dataset.map(one_hot_dataset)
        
    if take is not None:
        dataset = dataset.take(take)
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(take)) 
        

        
    dataset = dataset.shuffle(buffer_size=shuffle_buffer) if shuffle_buffer is not None else dataset
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset