from typing import Callable, List, Dict, Union, Optional
import tensorflow as tf


from jidenn.histogram.BinnedVariable import Binning

ROOTVariables = Dict[str, Union[tf.RaggedTensor, tf.Tensor]]
"""Type alias for a dictionary of ROOT variables. The keys are the variable  names and the values are the corresponding 
Tensorflow `tf.RaggedTensor` or `tf.Tensor`.

Example:
```python
variables = {
    'jets_pt': tf.RaggedTensor([[1, 2, 3, 4, 5], [2, 3]], dtype=tf.float32),
    'eventNumber': tf.Tensor([1, 2], dtype=tf.int32),
    ...
}
```
"""

def get_bin_idxing_fn(variable: str, bin_edges, sparse: bool = False) -> Callable[[ROOTVariables], tf.Tensor]:
    @tf.function
    def bin_idx(data: ROOTVariables) -> tf.Tensor:
        var = data[variable]
        var = tf.reshape(var, (1, ))
        var = tf.cast(var, tf.float32)
        bin_indices = tf.searchsorted(bin_edges, var, side='right') - 1
        bin_indices = tf.where(bin_indices < 0, 0, bin_indices)
        bin_indices = tf.clip_by_value(bin_indices, 0, len(bin_edges) - 2)
        bin_indices = tf.squeeze(bin_indices)
        if sparse:
            return bin_indices
        else:
            return tf.one_hot(bin_indices, len(bin_edges) - 1)
    return bin_idx


def get_reducing_fn(bin_idxing_fn: Callable[[ROOTVariables], tf.Tensor], weights: Optional[str] = None) -> Callable[[tf.Tensor, ROOTVariables], tf.Tensor]:
    @tf.function
    def reduce_fn(a: tf.Tensor, x: ROOTVariables) -> tf.Tensor:
        return a + bin_idxing_fn(x)

    if weights is None:
        return reduce_fn

    @tf.function
    def reduce_fn(a: tf.Tensor, x: ROOTVariables) -> tf.Tensor:
        return a + bin_idxing_fn(x) * tf.cast(x[weights], a.dtype)

    return reduce_fn


def get_reducing_fn_multiple(bin_idxing_fn: List[Callable[[ROOTVariables], tf.Tensor]], weights: Optional[str] = None) -> Callable[[List[tf.Tensor], ROOTVariables], List[tf.Tensor]]:
    @tf.function
    def reduce_fn(state: Dict[str, tf.Tensor], sample: ROOTVariables) -> List[tf.Tensor]:
        return {var: st + fn(sample) for (var, st), fn in zip(state.items(), bin_idxing_fn)}

    if weights is None:
        return reduce_fn

    @tf.function
    def reduce_fn(state: Dict[str, tf.Tensor], x: ROOTVariables) -> List[tf.Tensor]:
        return {var: st + fn(x) * tf.cast(x[weights], st.dtype) for (var, st), fn in zip(state.items(), bin_idxing_fn)}

    return reduce_fn


def histogram(dataset, binning: Binning, weight: Optional[str] = None):
    bin_idxing_fn = get_bin_idxing_fn(
        binning.variable, tf.cast(binning.bins, tf.float32))
    hist = tf.zeros((binning.n_bins, ), dtype=tf.float32)
    hist = dataset.reduce(hist, get_reducing_fn(bin_idxing_fn, weight))
    return hist


def histogram_multiple(dataset, binning: List[Binning], weight: Optional[str] = None):
    bin_idxing_fns = [get_bin_idxing_fn(
        b.variable, tf.cast(b.bins, tf.float32)) for b in binning]
    hists = {b.variable: tf.zeros(
        (b.n_bins, ), dtype=tf.float32) for b in binning}
    hists = dataset.reduce(
        hists, get_reducing_fn_multiple(bin_idxing_fns, weight))
    return hists
