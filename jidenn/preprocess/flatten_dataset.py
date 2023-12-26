"""
Set of functions to flatten a dataset. The flattening is done by the reference variable, which is assumed to be a
tf.RaggedTensor. The shape of the reference variable is used to infer the shape of the other variables. Th

"""
import tensorflow as tf
from typing import Tuple, List, Dict, Union, Optional, Callable

ROOTVariables = Dict[str, tf.RaggedTensor]


def get_ragged_to_dataset_fn(reference_variable: str = 'jets_PartonTruthLabelID',
                             key_phrase: str = 'jets') -> Callable[[ROOTVariables], tf.data.Dataset]:
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
        sample[f'{key_phrase}_index'] = tf.range(
            start=0, limit=ragged_shape[0], dtype=tf.int32)
        for key, item in sample.items():
            if key_phrase not in key or item.shape.num_elements() == 1:
                tensor_with_new_axis = tf.expand_dims(item, axis=0)
                sample[key] = tf.tile(tensor_with_new_axis, [
                                      ragged_shape[0]] + [1] * len(item.shape))
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
                                wanted_values: List[int] = [
                                    1, 2, 3, 4, 5, 6, 21],
                                key_phrase: str = 'jets') -> Callable[[ROOTVariables], ROOTVariables]:
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
        reference_tensor = sample[reference_variable]
        mask = tf.math.equal(tf.expand_dims(reference_tensor, axis=1), tf.tile(
            tf.expand_dims(wanted_values, axis=0), [tf.shape(reference_tensor)[0], 1]))
        mask = tf.reduce_any(mask, axis=1)
        for key, item in sample.items():
            if item.shape.num_elements() == 0 or item.shape.num_elements() == 1:
                continue
            if key_phrase in key:
                sample[key] = tf.ragged.boolean_mask(item, mask)
        return sample
    return _filter_unwanted_ragged_values_fn


def get_filter_ragged_cut(variable: str = 'jets_eta',
                          upper_cut: float = 2.5,
                          lower_cut: float = -2.5,
                          key_phrase: str = 'jets') -> Callable[[ROOTVariables], ROOTVariables]:
    """Get a function that filters out unwanted values from a ROOTVariables dictionary containg a RaggedTensor. The
    intended use is to use this function in a tf.data.Dataset.map call to filter out unwanted values from a dataset.
    the `get_filter_empty_fn` function should be used after this function to filter out empty events.

    Args:
        reference_variable (str, optional): The variable whose values to filter. Defaults to 'jets_PartonTruthLabelID'.
        wanted_values (List[int], optional): The values to keep. Defaults to [1, 2, 3, 4, 5, 6, 21].

    Returns:
        Callable[[ROOTVariables], ROOTVariables]: A function that filters out unwanted values from a ROOTVariables
    """
    # lower_cut = tf.broadcast_to(lower_cut, [])
    # if upper_cut is not None and lower_cut is not None:
    #     @tf.function
    #     def cut_fn(x):
    #         return
    # elif upper_cut is not None:
    #     @tf.function
    #     def cut_fn(x):
    #         return tf.math.greater(x, lower_cut)
    # elif lower_cut is not None:
    #     @tf.function
    #     def cut_fn(x):
    #         return tf.math.less(x, upper_cut)
    # else:
    #     raise ValueError('Both upper_cut and lower_cut cannot be None')

    @tf.function
    def _filter_unwanted_ragged_cut_fn(sample: ROOTVariables) -> ROOTVariables:
        sample = sample.copy()
        reference_tensor = sample[variable]
        mask = tf.math.logical_and(tf.math.greater(reference_tensor, lower_cut),
                                   tf.math.less(reference_tensor, upper_cut))
        for key, item in sample.items():
            if item.shape.num_elements() == 0 or item.shape.num_elements() == 1:
                continue
            if key_phrase in key:
                sample[key] = tf.ragged.boolean_mask(item, mask)
        return sample
    return _filter_unwanted_ragged_cut_fn


def get_keys_to_remove_fn(keys_to_remove: List[str]) -> Callable[[ROOTVariables], ROOTVariables]:
    """Get a function that removes keys from a ROOTVariables dictionary. The intended use is to use this function in a
    tf.data.Dataset.map call to remove unwanted keys from a dataset.

    Args:
        keys_to_remove (List[str]): The keys to remove.

    Returns:
        Callable[[ROOTVariables], ROOTVariables]: A function that removes keys from a ROOTVariables
    """
    @tf.function
    def _remove_keys(sample: ROOTVariables) -> ROOTVariables:
        sample = sample.copy()
        for key in keys_to_remove:
            sample.pop(key)
        return sample
    return _remove_keys


def flatten_dataset(dataset: tf.data.Dataset,
                    reference_variable: str = 'jets_PartonTruthLabelID',
                    wanted_values: Optional[List[int]] = None,
                    key_phrase: str = 'jets',
                    variables: Optional[Union[str, List[str]]] = None,
                    upper_cuts: Optional[Union[float, List[float]]] = None,
                    lower_cuts: Optional[Union[float, List[float]]] = None,
                    keys_to_remove: Optional[Union[List[str], str]] = ['jets_bTagged', 'jets_truth_flavor']) -> tf.data.Dataset:
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
    if keys_to_remove is not None:
        if isinstance(keys_to_remove, str):
            keys_to_remove = [keys_to_remove]
        dataset = dataset.map(get_keys_to_remove_fn(keys_to_remove))

    if wanted_values is None:
        return (
            dataset
            .filter(get_filter_empty_fn(reference_variable))
            .interleave(get_ragged_to_dataset_fn(reference_variable, key_phrase))
        )
    elif variables is None and upper_cuts is None and lower_cuts is None:
        return (
            dataset
            .map(get_filter_ragged_values_fn(reference_variable, wanted_values, key_phrase))
            .filter(get_filter_empty_fn(reference_variable))
            .interleave(get_ragged_to_dataset_fn(reference_variable, key_phrase))
        )
    else:
        if isinstance(variables, str):
            variables = [variables]
        if isinstance(upper_cuts, float) or isinstance(upper_cuts, int):
            upper_cuts = [upper_cuts]
        if isinstance(lower_cuts, float) or isinstance(lower_cuts, int):
            lower_cuts = [lower_cuts]

        for variable, upper_cut, lower_cut in zip(variables, upper_cuts, lower_cuts):
            dataset = dataset.map(get_filter_ragged_cut(
                variable, upper_cut, lower_cut, key_phrase))

        return (
            dataset
            .map(get_filter_ragged_values_fn(reference_variable, wanted_values, key_phrase))
            .filter(get_filter_empty_fn(reference_variable))
            .interleave(get_ragged_to_dataset_fn(reference_variable, key_phrase))
        )
