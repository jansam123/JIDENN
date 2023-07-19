import tensorflow as tf
import numpy as np
from typing import Callable, List, Tuple, Union, Optional

ROOTVariables = dict[str, Union[tf.Tensor, tf.RaggedTensor]]


@tf.function
def take_second(x, y):
    return y

@tf.function
def take_first(x, y):
    return x

def get_bin_fn(bins: Union[int, List[float]] = 100,
               lower_var_limit: int = 60_000,
               upper_var_limit: int = 5_600_000,
               log_binning_base: Optional[Union[int, float]] = None,
               variable: str = 'jets_pt') -> Callable[[ROOTVariables], tf.Tensor]:
    """Get a function that returns the bin index of a variable. The intended use is to use this function
    with tf.data.Dataset.rejection_resample() to resample a dataset to a target distribution.

    Args:
        n_bins (int, optional): Number of bins to use. Defaults to 100.
        lower_var_limit (int, optional): The lower limit of the variable. Defaults to 60_000.
        upper_var_limit (int, optional): The upper limit of the variable. Defaults to 5_600_000.
        variable (str, optional): Variable to bin. Defaults to 'jets_pt'.

    Returns:
        Callable[[ROOTVariables], tf.Tensor]: Function that returns the bin index of a variable.
    """
    
    if log_binning_base is not None and isinstance(bins, int):
        bin_edges =  np.logspace(np.log(lower_var_limit), np.log(upper_var_limit), bins + 1, base=log_binning_base)
        bin_edges = tf.constant(bin_edges, dtype=tf.float32)
    elif isinstance(bins, int):
        lower_var_limit_casted = tf.cast(lower_var_limit, dtype=tf.float32)
        upper_var_limit_casted = tf.cast(upper_var_limit, dtype=tf.float32)
        bin_edges = tf.linspace(lower_var_limit_casted, upper_var_limit_casted, bins + 1)
    else:
        bin_edges = tf.constant(bins, dtype=tf.float32)
        
    @tf.function
    def _rebin(data: ROOTVariables) -> tf.Tensor:
        var = data[variable]
        var = tf.reshape(var, (1, ))
        bin_indices = tf.searchsorted(bin_edges, var, side='right') - 1
        bin_indices = tf.where(bin_indices < 0, 0, bin_indices)
        bin_indices = tf.clip_by_value(bin_indices, 0, len(bin_edges) - 2)
        bin_indices = tf.squeeze(bin_indices)
        return bin_indices
    return _rebin


def resample_dataset(dataset: tf.data.Dataset,
                     bins: int = 100,
                     lower_var_limit: int = 60_000,
                     upper_var_limit: int = 5_600_000,
                     variable: str = 'jets_pt',
                     log_binning_base: Optional[Union[int, float]] = None,
                     target_dist: Optional[List[int]] = None,
                     precompute_init_dist: Optional[bool] = False) -> tf.data.Dataset:
    """Resample a dataset to a target distribution. 

    Args:
        dataset (tf.data.Dataset): The dataset to resample.
        n_bins (int, optional): Number of bins to use. Defaults to 100.
        lower_var_limit (int, optional): The lower limit of the binned variable (the lowest bin). 
            Defaults to 60_000.
        upper_var_limit (int, optional): The upper limit of the binned variable (the highest bin).
            Defaults to 5_600_000.
        variable (str, optional): Variable to bin. Defaults to 'jets_pt'.
        target_dist (Optional[List[int]], optional): The target distribution. If None, a uniform distribution
            is used. Defaults to None. It must have the same length as the number of bins.

    Returns:
        tf.data.Dataset: The resampled dataset.
    """

    dist = [1 / bins] * bins if target_dist is None else target_dist
    
    if target_dist is None and precompute_init_dist: 
        initial_dist = dataset.map(get_bin_fn(bins=bins,
                                            lower_var_limit=lower_var_limit,
                                            upper_var_limit=upper_var_limit,
                                            log_binning_base=log_binning_base,
                                            variable=variable))
        
        initial_dist = initial_dist.reduce(tf.zeros(bins, dtype=tf.int64), lambda x, y: x + tf.one_hot(y, bins, dtype=tf.int64))
        initial_dist = tf.cast(initial_dist, dtype=tf.float64)
        initial_dist = initial_dist / tf.reduce_sum(initial_dist)
        # initial_dist = tf.squeeze(initial_dist)
        print(initial_dist)
    else:
        initial_dist = None
    
    
    dataset = dataset.rejection_resample(get_bin_fn(bins=bins,
                                                    lower_var_limit=lower_var_limit,
                                                    log_binning_base=log_binning_base,
                                                    upper_var_limit=upper_var_limit,
                                                    variable=variable), target_dist=dist, initial_dist=initial_dist)
    return dataset.map(take_second)


def get_label_bin_fn(bins: int = 100,
                     lower_var_limit: Union[int, float] = 60_000,
                     upper_var_limit: Union[int, float] = 5_600_000,
                     variable: str = 'jets_pt',
                     label_variable: str = 'jets_PartonTruthLabelID',
                     label_class_1: List[int] = [1, 2, 3, 4, 5, 6],
                     log_binning_base: Optional[Union[int, float]] = None,
                     label_class_2: List[int] = [21]) -> Callable[[ROOTVariables], tf.Tensor]:
    """Get a function that returns the bin index of a data sample based on the value of the binned 
    variable and the label. The intended use is to use this function with tf.data.Dataset.rejection_resample()
    to resample a dataset to a target distribution.


    Args:
        n_bins (int, optional): Number of bins to use. Defaults to 100.
        lower_var_limit (Union[int, float], optional): The lower limit of the binned variable (the lowest bin). 
            Defaults to 60_000.
        upper_var_limit (Union[int, float], optional): The upper limit of the binned variable (the highest bin).
            Defaults to 5_600_000.
        variable (str, optional): Name of the variable to bin. Defaults to 'jets_pt'.
        label_variable (str, optional): Name of the label variable. Defaults to 'jets_PartonTruthLabelID'.
        label_class_1 (List[int], optional): Values of the label variable that correspond to the first class.
            The values will be regarded as the same class. Defaults to [1, 2, 3, 4, 5, 6].
        label_class_2 (List[int], optional): Values of the label variable that correspond to the second class.
            The values will be regarded as the same class. Defaults to [21].

    Returns:
        Callable[[ROOTVariables], tf.Tensor]: _description_
    """
    if log_binning_base is not None and isinstance(bins, int):
        bin_edges =  np.logspace(np.log(lower_var_limit), np.log(upper_var_limit), bins + 1, base=log_binning_base)
        bin_edges = tf.constant(bin_edges, dtype=tf.float32)
    elif isinstance(bins, int):
        lower_var_limit_casted = tf.cast(lower_var_limit, dtype=tf.float32)
        upper_var_limit_casted = tf.cast(upper_var_limit, dtype=tf.float32)
        bin_edges = tf.linspace(lower_var_limit_casted, upper_var_limit_casted, bins + 1)
    else:
        bin_edges = tf.constant(bins, dtype=tf.float32)
        
    n_bins = len(bin_edges) - 1
        
    @tf.function
    def _rebin(data: ROOTVariables) -> tf.Tensor:
        var = data[variable]
        label = data[label_variable]
        var = tf.reshape(var, (1, ))
        bin_indices = tf.searchsorted(bin_edges, var, side='right') - 1
        # bin_indices = tf.where(bin_indices < 0, 0, bin_indices)
        bin_indices = tf.clip_by_value(bin_indices, 0, n_bins - 1)
        bin_indices = tf.squeeze(bin_indices)
        if tf.reduce_any(tf.equal(label, tf.constant(label_class_1, dtype=label.dtype))):
            return bin_indices
        elif tf.reduce_any(tf.equal(label, tf.constant(label_class_2, dtype=label.dtype))):
            return bin_indices + n_bins
        else:
            return tf.constant(0, dtype=tf.int32)
    return _rebin


def rasample_from_min_bin_count(dataset: tf.data.Dataset,
                                bin_fn: Callable[[ROOTVariables], tf.Tensor],
                                min_bin_count: int,
                                bins: int = 100,) -> tf.data.Dataset:
    
    @tf.function
    def assign_if_filter_fn(state, x):
        bin_idx = bin_fn(x)
        state = state + tf.one_hot(bin_idx, bins, dtype=tf.int64)
        return state, (x, tf.greater(min_bin_count, state[bin_idx]))

    dataset = dataset.scan(tf.zeros((bins), dtype=tf.int64), assign_if_filter_fn)
    dataset = dataset.filter(take_second).map(take_first)
    
    return dataset
                                


def resample_with_labels_dataset(dataset: tf.data.Dataset,
                                 bins: int = 100,
                                 lower_var_limit: int = 60_000,
                                 upper_var_limit: int = 5_600_000,
                                 variable: str = 'jets_pt',
                                 label_variable: str = 'jets_PartonTruthLabelID',
                                 label_class_1: List[int] = [1, 2, 3, 4, 5, 6],
                                 label_class_2: List[int] = [21],
                                 target_dist: Optional[List[int]] = None,
                                 log_binning_base: Optional[Union[int, float]] = None,
                                 precompute_init_dist: Optional[bool] = False,
                                 from_min_count: bool = False) -> tf.data.Dataset:
    """Resample a dataset to a target distribution based on the value of the label variable and the binned variable.


    Args:
        dataset (tf.data.Dataset): The dataset to resample.
        n_bins (int, optional): Number of bins to use. Defaults to 100.
        lower_var_limit (Union[int, float], optional): The lower limit of the binned variable (the lowest bin). 
            Defaults to 60_000.
        upper_var_limit (Union[int, float], optional): The upper limit of the binned variable (the highest bin).
            Defaults to 5_600_000.
        variable (str, optional): Name of the variable to bin. Defaults to 'jets_pt'.
        label_variable (str, optional): Name of the label variable. Defaults to 'jets_PartonTruthLabelID'.
        label_class_1 (List[int], optional): Values of the label variable that correspond to the first class.
            The values will be regarded as the same class. Defaults to [1, 2, 3, 4, 5, 6].
        label_class_2 (List[int], optional): Values of the label variable that correspond to the second class.
            The values will be regarded as the same class. Defaults to [21].
        target_dist (Optional[List[int]], optional): The target distribution. If None, a uniform distribution
            is used. Defaults to None. It must have the same length as the number of bins.

    Returns:
        tf.data.Dataset: The resampled dataset.
    """
    dist = [1 / (bins * 2)] * bins * 2 if target_dist is None else target_dist
    
    if target_dist is None and precompute_init_dist: 
        initial_dist = dataset.map(get_label_bin_fn(bins=bins,
                                                            lower_var_limit=lower_var_limit,
                                                            upper_var_limit=upper_var_limit,
                                                            variable=variable,
                                                            label_variable=label_variable,
                                                            log_binning_base=log_binning_base,
                                                            label_class_1=label_class_1,
                                                            label_class_2=label_class_2))
        initial_dist = initial_dist.reduce(tf.zeros((bins * 2), dtype=tf.int64), lambda x, y: x + tf.one_hot(y, bins * 2, dtype=tf.int64))
        min_count = tf.reduce_min(initial_dist).numpy()
        min_count = tf.cast(min_count, dtype=tf.int64)
        initial_dist = tf.cast(initial_dist, dtype=tf.float64)
        print(f'Min count: {min_count:,}')
        print(f'Estimated number of samples: {min_count * bins * 2:,}')
        initial_dist = initial_dist / tf.reduce_sum(initial_dist)
        print(initial_dist)
    else:
        initial_dist = None
        
    label_bin_fn = get_label_bin_fn(bins=bins,
                                    lower_var_limit=lower_var_limit,
                                    upper_var_limit=upper_var_limit,
                                    variable=variable,
                                    log_binning_base=log_binning_base,
                                    label_variable=label_variable,
                                    label_class_1=label_class_1,
                                    label_class_2=label_class_2)
        
    if from_min_count:
        return rasample_from_min_bin_count(dataset, label_bin_fn, min_bin_count=min_count, bins=bins * 2)
    else:
        dataset = dataset.rejection_resample(label_bin_fn, target_dist=dist, initial_dist=initial_dist)
        return dataset.map(take_second)


def write_new_variable(variable_value: int, variable_name: str = 'JZ_slice') -> Callable[[ROOTVariables], ROOTVariables]:
    """Get a function that writes a new variable to a dataset. The intended use is to use this function
    with tf.data.Dataset.map() to add a new variable to a dataset. The value is the same for all samples.

    Args:
        variable_value (int): The value of the new variable.
        variable_name (str, optional): The name of the new variable. Defaults to 'JZ_slice'.

    Returns:
        Callable[[ROOTVariables], ROOTVariables]: Function that writes a new variable to a dataset.
    """

    @tf.function
    def _write(data: ROOTVariables) -> ROOTVariables:
        new_data = data.copy()
        new_data[variable_name] = variable_value
        return new_data
    return _write


def get_label_splitting_fn(label_variable: str = 'jets_PartonTruthLabelID',
                           label_class_1: List[int] = [1, 2, 3, 4, 5, 6],
                           label_class_2: List[int] = [21]) -> Callable[[ROOTVariables], tf.Tensor]:
    """Get a function that splits a dataset based on the value of the label variable. The intended use is to use this function
    with tf.data.Dataset.filter() to split a dataset into two classes.

    Args:
        label_variable (str, optional): Name of the label variable. Defaults to 'jets_PartonTruthLabelID'.
        label_class_1 (List[int], optional): Values of the label variable that correspond to the first class.
        label_class_2 (List[int], optional): Values of the label variable that correspond to the second class.

    Returns:
        Callable[[ROOTVariables], tf.Tensor]: Function that splits a dataset based on the value of the label variable.
    """
    @tf.function
    def _split(data: ROOTVariables) -> tf.Tensor:
        label = data[label_variable]
        if tf.reduce_any(tf.equal(label, tf.constant(label_class_1, dtype=tf.int32))):
            return tf.constant(1, dtype=tf.int32)
        elif tf.reduce_any(tf.equal(label, tf.constant(label_class_2, dtype=tf.int32))):
            return tf.constant(0, dtype=tf.int32)
        else:
            return tf.constant(0, dtype=tf.int32)
    return _split


def get_cut_fn(variable: str = 'jets_pt', lower_limit: Optional[float] = None, upper_limit: Optional[float] = None) -> Callable[[ROOTVariables], tf.Tensor]:
    """Get a function that cuts a dataset based on the value of a variable. The intended use is to use this function
    with tf.data.Dataset.filter() to cut a dataset.

    Args:
        variable (str, optional): Name of the variable to cut. Defaults to 'jets_pt'.
        lower_limit (Optional[float], optional): The lower limit of the variable. Defaults to None.
        upper_limit (Optional[float], optional): The upper limit of the variable. Defaults to None.

    Raises:
        ValueError: At least one of the limits must be specified.

    Returns:
        Callable[[ROOTVariables], tf.Tensor]: Function that cuts a dataset based on the value of a variable.
    """

    if lower_limit is None and upper_limit is None:
        raise ValueError('At least one of the limits must be specified.')

    @tf.function
    def _low_cut(data: ROOTVariables) -> tf.Tensor:
        return tf.reduce_all(data[variable] > lower_limit)

    @tf.function
    def _up_cut(data: ROOTVariables) -> tf.Tensor:
        return tf.reduce_all(data[variable] < upper_limit)

    @tf.function
    def _up_low_cut(data: ROOTVariables) -> tf.Tensor:
        return tf.reduce_all([data[variable] > lower_limit, data[variable] < upper_limit])

    if lower_limit is None:
        return _up_cut
    elif upper_limit is None:
        return _low_cut
    else:
        return _up_low_cut


def get_filter_fn(variable: str, values: List[Union[float, int]]) -> Callable[[ROOTVariables], tf.Tensor]:
    """Get a function that filters a dataset based on the value of a variable. The intended use is to use this function
    with tf.data.Dataset.filter() to filter a dataset.

    Args:
        variable (str): Name of the variable to filter.
        values (List[Union[float, int]]): Values to keep.

    Returns:
        Callable[[ROOTVariables], tf.Tensor]: Function that filters a dataset based on the value of a variable.
    """
    @tf.function
    def _filter(data: ROOTVariables) -> tf.Tensor:
        return tf.reduce_any(tf.equal(data[variable], tf.constant(values)))
    return _filter
