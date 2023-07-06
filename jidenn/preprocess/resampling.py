import tensorflow as tf
from typing import Callable, List, Tuple, Union, Optional

ROOTVariables = dict[str, Union[tf.Tensor, tf.RaggedTensor]]


def get_bin_fn(n_bins: int = 100,
               lower_var_limit: int = 60_000,
               upper_var_limit: int = 5_600_000,
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
    @tf.function
    def _rebin(data: ROOTVariables) -> tf.Tensor:
        var = data[variable]
        var = tf.reshape(var, ())
        index = tf.histogram_fixed_width_bins(var, (lower_var_limit, upper_var_limit), nbins=n_bins)
        return index
    return _rebin


def resample_dataset(dataset: tf.data.Dataset,
                     n_bins: int = 100,
                     lower_var_limit: int = 60_000,
                     upper_var_limit: int = 5_600_000,
                     variable: str = 'jets_pt',
                     target_dist: Optional[List[int]] = None) -> tf.data.Dataset:
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

    dist = [1 / (n_bins * 2)] * n_bins * 2 if target_dist is None else target_dist
    dataset = dataset.rejection_resample(get_bin_fn(n_bins=n_bins,
                                                    lower_var_limit=lower_var_limit,
                                                    upper_var_limit=upper_var_limit,
                                                    variable=variable), target_dist=dist)
    @tf.function
    def take_second(x, y):
        return y
    
    return dataset.map(take_second)


def get_label_bin_fn(n_bins: int = 100,
                     lower_var_limit: Union[int, float] = 60_000,
                     upper_var_limit: Union[int, float] = 5_600_000,
                     variable: str = 'jets_pt',
                     label_variable: str = 'jets_PartonTruthLabelID',
                     label_class_1: List[int] = [1, 2, 3, 4, 5, 6],
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
    @tf.function
    def _rebin(data: ROOTVariables) -> tf.Tensor:
        var = data[variable]
        var = tf.reshape(var, ())
        lower_var_limit_casted = tf.cast(lower_var_limit, dtype=var.dtype)
        upper_var_limit_casted = tf.cast(upper_var_limit, dtype=var.dtype)
        index = tf.histogram_fixed_width_bins(var, (lower_var_limit_casted, upper_var_limit_casted), nbins=n_bins)
        label = data[label_variable]
        if tf.reduce_any(tf.equal(label, tf.constant(label_class_1, dtype=label.dtype))):
            return index
        elif tf.reduce_any(tf.equal(label, tf.constant(label_class_2, dtype=label.dtype))):
            return index + n_bins
        else:
            return tf.constant(0, dtype=tf.int32)
    return _rebin


def resample_with_labels_dataset(dataset: tf.data.Dataset,
                                 n_bins: int = 100,
                                 lower_var_limit: int = 60_000,
                                 upper_var_limit: int = 5_600_000,
                                 variable: str = 'jets_pt',
                                 label_variable: str = 'jets_PartonTruthLabelID',
                                 label_class_1: List[int] = [1, 2, 3, 4, 5, 6],
                                 label_class_2: List[int] = [21],
                                 target_dist: Optional[List[int]] = None) -> tf.data.Dataset:
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

    dist = [1 / (n_bins * 2)] * n_bins * 2 if target_dist is None else target_dist
    dataset = dataset.rejection_resample(get_label_bin_fn(n_bins=n_bins,
                                                          lower_var_limit=lower_var_limit,
                                                          upper_var_limit=upper_var_limit,
                                                          variable=variable,
                                                          label_variable=label_variable,
                                                          label_class_1=label_class_1,
                                                          label_class_2=label_class_2), target_dist=dist)
    
    @tf.function
    def take_second(x, y):
        return y
    
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
