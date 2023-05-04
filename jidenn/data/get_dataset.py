"""
Module containing a single function to load and preprocess a dataset from a list of files 
into `JIDENNDataset` objects, perform the preprocessing steps specified in `args_data`.
"""
import tensorflow as tf
from typing import List, Optional, Tuple, Callable

from jidenn.config import config
from jidenn.data.string_conversions import Cut
from jidenn.data.JIDENNDataset import JIDENNDataset, ROOTVariables


def get_preprocessed_dataset(files: List[str],
                             args_data: config.Data,
                             files_labels: Optional[List[int]] = None) -> JIDENNDataset:
    """Loads and preprocesses a dataset from a list of files into `JIDENNDataset` objects, performs the preprocessing steps specified in `args_data` 
    and combines the datasets into one. 

    All preprocessing steps are tailored to the dataset used in quark/gluon tagging:

    1. Loop over all `files` and  `files_labels`.
    2. Create the `JIDENNDataset` object from the `args_data` configuration.
    3. Load the dataset from file into a single `JIDENNDataset` object.
    4. Apply the `JZ_cut` and `cut` specified in `args_data` to the dataset.
    5. Resample the dataset to a 50/50 gluon/quark ratio if `resample_labels` is `True`.
    6. Create a variable in the dataset containing the `files_labels` stamp.
    7. Remap the labels to gluon = 0 and quarks = 1.
    8. Combine all datasets into one.

    Args:
        files: A list of strings containing the file paths of the datasets to be loaded. Each file path points to a dataset saved with tf.data.experimental.save.
        args_data: Data configuration containing the required preprocessing settings.
        files_labels: A list of strings containing the stamps of the datasets to be loaded. Each stamp is used to identify the datafile it belongs to.

    Returns:
        A JIDENNDataset object containing the preprocessed dataset.
    """

    var_labels_1 = tf.constant(args_data.target_labels[0], dtype=tf.int32)
    var_labels_2 = tf.constant(args_data.target_labels[1], dtype=tf.int32)

    @tf.function
    def resample(d: ROOTVariables, x: tf.Tensor) -> int:
        if x.dtype != tf.int32:
            x = tf.cast(x, tf.int32)
        if tf.reduce_any(x == var_labels_1):
            return 0
        elif tf.reduce_any(x == var_labels_2):
            return 1
        else:
            return -999

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

    def stamp_origin_file(stamp: int) -> Callable[[ROOTVariables], ROOTVariables]:
        @tf.function
        def stamp_origin_file_wrap(data: ROOTVariables) -> ROOTVariables:
            new_data = data.copy()
            new_data['origin_file'] = tf.constant(stamp, tf.int32)
            return new_data
        return stamp_origin_file_wrap

    @tf.function
    def count_PFO(sample: ROOTVariables) -> ROOTVariables:
        for pfo_var in ['jets_PFO_pt', 'jets_PFO_eta', 'jets_PFO_phi', 'jets_PFO_m']:
            if pfo_var in sample.keys():
                sample = sample.copy()
                sample['jets_PFO_n'] = tf.reduce_sum(
                    tf.ones_like(sample['jets_PFO_pt']))
                break
        return sample

    JZ_cuts = args_data.subfolder_cut if args_data.subfolder_cut is not None else [
        None] * len(files)

    datasets = []
    for i, (jz_cut, jz_file) in enumerate(zip(JZ_cuts, files)):

        jidenn_dataset = JIDENNDataset(variables=args_data.variables,
                                       target=args_data.target,
                                       weight=args_data.weight)
        jidenn_dataset = jidenn_dataset.load_dataset(jz_file)

        if jz_cut is not None:
            cut = Cut(jz_cut) & Cut(
                args_data.cut) if args_data.cut is not None else Cut(jz_cut)
        elif args_data.cut is not None:
            cut = Cut(args_data.cut)
        else:
            cut = None

        jidenn_dataset = jidenn_dataset.create_variables(
            cut=cut, map_dataset=count_PFO)

        if args_data.resample_labels is not None:
            if len(args_data.target_labels) != len(args_data.resample_labels):
                raise ValueError(
                    'The number of target labels and resample labels must be the same.')
            jidenn_dataset = jidenn_dataset.resample_dataset(
                resample, args_data.resample_labels)

        jidenn_dataset = jidenn_dataset.create_train_input(stamp_origin_file(
            files_labels[i])) if files_labels is not None else jidenn_dataset

        jidenn_dataset = jidenn_dataset.remap_labels(label_mapping)
        datasets.append(jidenn_dataset)

    if len(datasets) == 1:
        return datasets[0]

    return JIDENNDataset.combine(datasets, args_data.subfolder_weights)



