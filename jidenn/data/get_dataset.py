"""
Module containing a single function to load and preprocess a dataset from a list of files 
into `JIDENNDataset` objects, perform the preprocessing steps specified in `args_data`.
"""
import tensorflow as tf
from typing import List, Optional, Tuple, Callable, Union

from jidenn.config import config
from jidenn.data.string_conversions import Cut
from jidenn.data.JIDENNDataset import JIDENNDataset, ROOTVariables


def get_preprocessed_dataset(args_data: config.Data,
                             input_creator: Callable[[ROOTVariables], ROOTVariables],
                             augmentation: Optional[Callable[[ROOTVariables], ROOTVariables]] = None,
                             shuffle_reading: bool = False):

    @tf.function
    def count_PFO(sample: ROOTVariables) -> ROOTVariables:
        for pfo_var in ['jets_PFO_pt', 'jets_PFO_eta', 'jets_PFO_phi', 'jets_PFO_m']:
            if pfo_var in sample.keys():
                sample = sample.copy()
                sample['jets_PFO_n'] = tf.reduce_sum(
                    tf.ones_like(sample['jets_PFO_pt']))
                break
        return sample

    var_labels_1 = tf.constant(args_data.target_labels[0], dtype=tf.int32)
    var_labels_2 = tf.constant(args_data.target_labels[1], dtype=tf.int32)
    unknown_labels = tf.constant(args_data.variable_unknown_labels, dtype=tf.int32)

    @tf.function
    def label_mapping(x: int) -> int:
        if x.dtype != tf.int32:
            x = tf.cast(x, tf.int32)
        if tf.reduce_any(x == var_labels_1):
            return 0
        elif tf.reduce_any(x == var_labels_2):
            return 1
        else:
            return -999

    @tf.function
    def filter_unknown_labels(sample: ROOTVariables) -> bool:
        is_unknown = tf.reduce_any(sample[args_data.target] == unknown_labels)
        return tf.logical_not(is_unknown)
    
        
    @tf.function
    def reweight_dataset(sample: ROOTVariables, file_label) -> ROOTVariables:
        sample = sample.copy()
        sample[args_data.weight] = sample[args_data.weight] / args_data.dataset_norm[file_label] 
        return sample

    if isinstance(args_data.path, str):
        dataset = JIDENNDataset.load(args_data.path, shuffle_reading=shuffle_reading)
    else:
        file_labels = list(range(len(args_data.path)))
        dataset = JIDENNDataset.load_multiple(args_data.path, 
                                              mode='sample', 
                                              weights=args_data.dataset_weigths, 
                                              stop_on_empty_dataset=True, 
                                              only_common_variables=True,
                                              data_mapper=reweight_dataset if args_data.dataset_norm is not None else None,
                                              file_labels=file_labels if args_data.dataset_norm else None,)
    dataset = dataset.remap_data(count_PFO)
    dataset = dataset.filter(Cut(args_data.cut)) if args_data.cut is not None else dataset
    dataset = dataset.filter(filter_unknown_labels) if args_data.variable_unknown_labels is not None else dataset
    dataset = dataset.remap_data(augmentation) if augmentation is not None else dataset
    dataset = dataset.set_variables_target_weight(target=args_data.target, weight=args_data.weight)
    dataset = dataset.remap_labels(label_mapping)
    dataset = dataset.remap_data(input_creator) if input_creator is not None else dataset
    return dataset



