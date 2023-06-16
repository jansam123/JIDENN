import tensorflow as tf
from typing import Callable, List, Tuple, Union, Optional

ROOTVariables = dict[str, Union[tf.Tensor, tf.RaggedTensor]]


def get_bin_fn(n_bins: int = 100,
               lower_pt_limit: int = 60_000,
               upper_pt_limit: int = 5_600_000,
               variable: str = 'jets_pt') -> Callable[[ROOTVariables], tf.Tensor]:
    @tf.function
    def _rebin(data: ROOTVariables) -> tf.Tensor:
        var = data[variable]
        var = tf.reshape(var, ())
        index = tf.histogram_fixed_width_bins(var, (lower_pt_limit, upper_pt_limit), nbins=n_bins)
        return index
    return _rebin


def get_label_bin_fn(n_bins: int = 100,
                     lower_pt_limit: int = 60_000,
                     upper_pt_limit: int = 5_600_000,
                     variable: str = 'jets_pt',
                     label_variable: str = 'jets_PartonTruthLabelID',
                     label_class_1: List[int] = [1, 2, 3, 4, 5, 6],
                     label_class_2: List[int] = [21]) -> Callable[[ROOTVariables], tf.Tensor]:
    @tf.function
    def _rebin(data: ROOTVariables) -> tf.Tensor:
        var = data[variable]
        var = tf.reshape(var, ())
        index = tf.histogram_fixed_width_bins(var, (lower_pt_limit, upper_pt_limit), nbins=n_bins)
        label = data[label_variable]
        if tf.reduce_any(tf.equal(label, tf.constant(label_class_1, dtype=tf.int32))):
            return index
        elif tf.reduce_any(tf.equal(label, tf.constant(label_class_2, dtype=tf.int32))):
            return index + n_bins
        else:
            return tf.constant(0, dtype=tf.int32)
    return _rebin


def write_new_variable(variable_value: int, variable_name: str = 'JZ_slice') -> Callable[[ROOTVariables], ROOTVariables]:
    @tf.function
    def _write(data: ROOTVariables) -> ROOTVariables:
        new_data = data.copy()
        new_data[variable_name] = variable_value
        return new_data
    return _write


def get_label_splitting_fn(label_variable: str = 'jets_PartonTruthLabelID',
                           label_class_1: List[int] = [1, 2, 3, 4, 5, 6],
                           label_class_2: List[int] = [21]) -> Callable[[ROOTVariables], tf.Tensor]:
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

    if lower_limit is None and upper_limit is None:
        raise ValueError('At least one of the limits must be specified.')

    @tf.function
    def _low_cut(data):
        return tf.reduce_all(data[variable] > lower_limit)

    @tf.function
    def _up_cut(data):
        return tf.reduce_all(data[variable] < upper_limit)

    @tf.function
    def _up_low_cut(data):
        return tf.reduce_all([data[variable] > lower_limit, data[variable] < upper_limit])

    if lower_limit is None:
        return _up_cut
    elif upper_limit is None:
        return _low_cut
    else:
        return _up_low_cut


def get_filter_fn(variable: str, values: List[Union[float, int]]):
    @tf.function
    def _filter(data):
        return tf.reduce_any(tf.equal(data[variable], tf.constant(values)))
    return _filter
