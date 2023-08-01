"""
Set of functions to flatten a dataset. The flattening is done by the reference variable, which is assumed to be a
tf.RaggedTensor. The shape of the reference variable is used to infer the shape of the other variables. Th

"""
import tensorflow as tf
from typing import Tuple, List, Dict, Union, Optional, Callable

ROOTVariables = Dict[str, tf.RaggedTensor]


def get_ragged_to_dataset_fn(reference_variable: str = 'jets_PartonTruthLabelID') -> Callable[[ROOTVariables], tf.data.Dataset]:
    """Get a function that converts a tf.RaggedTensor to a tf.data.Dataset. The intended use is to use this function 
    in a tf.data.Dataset.interleave call to flatten a dataset. The function will infer the shape of the ragged tensor
    from the shape of the reference variable. Variables toher than the reference variable will be tiled to match the
    shape of the reference variable (they will be duplicated).

    Args:
        reference_variable (str, optional): The variable to use as reference for infering the shape of
            variables to flatten. Defaults to 'jets_PartonTruthLabelID'.

    Returns:
        Callable[[ROOTVariables], tf.data.Dataset]: A function that converts a tf.RaggedTensor to a tf.data.Dataset
    """
    @tf.function
    def _ragged_to_dataset(sample: ROOTVariables) -> tf.data.Dataset:
        sample = sample.copy()
        ragged_shape = tf.shape(sample[reference_variable])
        for key, item in sample.items():
            if isinstance(item, tf.RaggedTensor) and ragged_shape[0] == tf.shape(item)[0]:
                continue
            elif len(tf.shape(item)) == 0:
                sample[key] = tf.tile(item[tf.newaxis, tf.newaxis], [ragged_shape[0], 1])
            elif tf.shape(item)[0] != ragged_shape[0]:
                sample[key] = tf.tile(item[tf.newaxis, :], [ragged_shape[0], 1])
            else:
                continue
        return tf.data.Dataset.from_tensor_slices(sample)
    return _ragged_to_dataset


def get_filter_empty_fn(reference_variable: str = 'jets_PartonTruthLabelID') -> Callable[[ROOTVariables], tf.Tensor]:
    """Get a function that filters out empty RaggedTensors from a ROOTVariables dictionary. The intended use is to use
    this function in a tf.data.Dataset.filter call to filter out empty events.

    Args:
        reference_variable (str, optional): The variable to use as reference for infering the shape of
            empty events. Defaults to 'jets_PartonTruthLabelID'.

    Returns:
        Callable[[ROOTVariables], tf.Tensor]: A function that filters out empty RaggedTensors from a ROOTVariables
    """
    @tf.function
    def _filter_empty(sample: ROOTVariables) -> tf.Tensor:
        return tf.greater(tf.size(sample[reference_variable]), 0)
    return _filter_empty


def get_filter_ragged_values_fn(reference_variable: str = 'jets_PartonTruthLabelID',
                                         wanted_values: List[int] = [1, 2, 3, 4, 5, 6, 21]) -> Callable[[ROOTVariables], ROOTVariables]:
    """Get a function that filters out unwanted values from a ROOTVariables dictionary containg a RaggedTensor. The
    intended use is to use this function in a tf.data.Dataset.map call to filter out unwanted values from a dataset.
    the `get_filter_empty_fn` function should be used after this function to filter out empty events.

    Args:
        reference_variable (str, optional): The variable whose values to filter. Defaults to 'jets_PartonTruthLabelID'.
        wanted_values (List[int], optional): The values to keep. Defaults to [1, 2, 3, 4, 5, 6, 21].

    Returns:
        Callable[[ROOTVariables], ROOTVariables]: A function that filters out unwanted values from a ROOTVariables
    """
    @tf.function
    def _filter_unwanted_ragged_values_fn(sample: ROOTVariables) -> ROOTVariables:
        sample = sample.copy()
        mask = tf.math.reduce_any(tf.math.equal(sample[reference_variable], wanted_values), axis=-1)
        for key, item in sample.items():
            if tf.reduce_all(tf.math.equal(tf.shape(item), tf.shape(mask))):
                sample[key] = tf.ragged.boolean_mask(item, mask)
        return sample
    return _filter_unwanted_ragged_values_fn


def flatten_dataset(dataset: tf.data.Dataset,
                    reference_variable: str = 'jets_PartonTruthLabelID',
                    wanted_values: Optional[List[int]] = None) -> tf.data.Dataset:
    """Apply a series of transformations to a tf.data.Dataset to flatten it. The flattening is done by the reference
    variable, which is assumed to be a tf.RaggedTensor. The shape of the reference variable is used to infer the shape
    of the other variables. The other variables are tiled to match the shape of the reference variable. The dataset is
    then filtered to remove empty events. If wanted_values is not None, the reference variable is filtered to only
    contain the wanted values.

    Args:
        dataset (tf.data.Dataset): The dataset to flatten.
        reference_variable (str, optional): The variable to use as reference for infering the shape of
            variables to flatten. Defaults to 'jets_PartonTruthLabelID'.
        wanted_values (Optional[List[int]], optional): The values to keep in the reference variable. Defaults to None.

    Returns:
        tf.data.Dataset: The flattened dataset
    """

    if wanted_values is None:
        return (
            dataset
            .map(get_filter_empty_fn(reference_variable))
            .interleave(get_ragged_to_dataset_fn(reference_variable))
        )
    else:
        return (
            dataset
            .map(get_filter_ragged_values_fn(reference_variable, wanted_values))
            .map(get_filter_empty_fn(reference_variable))
            .interleave(get_ragged_to_dataset_fn(reference_variable))
        )