from typing import Callable
from .JIDENNDataset import JIDENNDataset
from src.config import config_subclasses as cfg        
from src.data.preprocess import pipe
import tensorflow as tf
        
        
def get_dataset(files:list[str],
                args_data:cfg.Data, 
                filter: Callable | None = None) -> tf.data.Dataset:
    
    dataset = JIDENNDataset(files=files,
                            variables=args_data.variables,
                            target=args_data.target,
                            weight=args_data.weight,
                            reading_size=args_data.reading_size,
                            num_workers=args_data.num_workers,
                            cut=args_data.cut,).dataset
    
    dataset = dataset.filter(filter) if filter is not None else dataset
    return dataset        
                         
def get_preprocessed_dataset(files: list[list[str]],
                   args_data: cfg.Data,
                   args_dataset: cfg.Dataset,
                   name:str,
                   size: int | None = None) -> tf.data.Dataset:
    
    quarks_tensor = tf.constant(args_data.raw_quarks)
    @tf.function
    def filter_mixed(x,y,z):
        return y == args_data.raw_gluon or tf.reduce_any(tf.equal(y, quarks_tensor))
    @tf.function
    def filter_quarks(x,y,z):
        return tf.reduce_any(tf.equal(y, quarks_tensor))
    @tf.function
    def filter_gluons(x,y,z):
        return y == args_data.raw_gluon
    
    datasets = []
    for sub_files in files:
        gluon_dataset = get_dataset(sub_files, args_data, filter_gluons)
        quark_dataset = get_dataset(sub_files, args_data, filter_quarks)
        datasets.append(tf.data.Dataset.sample_from_datasets([gluon_dataset, quark_dataset], [0.5, 0.5], stop_on_empty_dataset=True))
        
    if len(files) == 1:
        dataset = datasets[0]
    else:
        dataset = tf.data.Dataset.sample_from_datasets(datasets, [1/len(datasets)]*len(datasets), stop_on_empty_dataset=True)
    
    @tf.function
    def label_mapping(x):
        if x == args_data.raw_gluon:
            return args_data.gluon
        else:
            return args_data.quark
        
    return pipe(dataset, args_dataset, label_mapping=label_mapping, name=name, take=size)