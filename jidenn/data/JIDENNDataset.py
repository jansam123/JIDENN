"""
Module containing the `JIDENNDataset` dataclass that is a wrapper for a TensorFlow dataset that allows for easy adding and processing of dataset files.
It contains all the necessary tools to perform a preprocessing of the jet dataset for training. 
"""
from __future__ import annotations
import tensorflow as tf
import pandas as pd
from dataclasses import dataclass
from typing import Union, Literal, Callable, Dict, Tuple, List, Optional, Any, Iterator
import os
import pickle
import awkward as ak
import uproot
import logging
import numpy as np
import ray
#
from jidenn.data.string_conversions import Cut, Expression
from jidenn.evaluation.plotter import plot_data_distributions, plot_single_dist
from jidenn.histogram.tf_dataset_histogram import histogram, histogram_multiple
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
FLOAT_PRECISION = tf.float32
INT_PRECISION = tf.int32


def all_equal(iterator):
    """Checks if all elements in an iterator are equal."""
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == x for x in iterator)


def pandas_to_tensor(df: pd.Series) -> Union[tf.RaggedTensor, tf.Tensor]:
    """Converts a pandas `pd.Series` to a Tensorflow `tf.RaggedTensor` or `tf.Tensor`. The output is a `tf.RaggedTensor`
    if the Series has a multiple level index, otherwise it is a `tf.Tensor`. The number of levels of the index gives the
    number of dimensions of the output. 

    Args:
        df (pd.Series): pandas pd.Series to be converted. Can have a single or multiple level index (`pd.MultiIndex`).

    Returns:
        tf.RaggedTensor or tf.Tensor: `tf.RaggedTensor` if df has number of index levels  greater than 1, else `tf.Tensor`.
    """
    levels = df.index.nlevels
    if levels == 1:
        return tf.constant(df.values)
    elif levels == 2:
        row_lengths = df.groupby(level=[0]).count()
        return tf.RaggedTensor.from_row_lengths(df.values, row_lengths.values, validate=False)
    else:
        max_level_group = list(range(levels - 1))
        nested_row_lengths = [df.groupby(level=max_level_group).count()]
        for i in range(1, levels - 1):
            nested_row_lengths.append(
                nested_row_lengths[-1].groupby(level=max_level_group[:-i]).count())
        return tf.RaggedTensor.from_nested_row_lengths(df.values, nested_row_lengths=nested_row_lengths[::-1], validate=False)


def awkward_to_tensor(array: ak.Array) -> Union[tf.RaggedTensor, tf.Tensor]:
    """Converts an awkward `ak.Array` to a Tensorflow `tf.RaggedTensor` or tf.Tensor. The output is a `tf.RaggedTensor` 
    if the array has a dimension greater than 1, otherwise it is a `tf.Tensor`. The number of dimensions of the array 
    gives the number of dimensions of the output.

    Args:
        array (ak.Array): awkward ak.Array to be converted. Can have a single or multiple dimensions.

    Returns:
        tf.RaggedTensor or tf.Tensor: `tf.RaggedTensor` if the array dimension is greater than 1, else `tf.Tensor`.
    """
    if array.ndim == 1:
        return tf.constant(array.to_list())
    elif array.ndim == 2:
        row_lengths = ak.num(array, axis=1).to_list()
        return tf.RaggedTensor.from_row_lengths(ak.flatten(array, axis=None).to_list(), row_lengths=row_lengths, validate=False)
    else:
        nested_row_lengths = [ak.flatten(ak.num(array, axis=ax), axis=None).to_list()
                              for ax in range(1, array.ndim)]
        return tf.RaggedTensor.from_nested_row_lengths(ak.flatten(
            array, axis=None).to_list(), nested_row_lengths=nested_row_lengths, validate=False)


def read_ttree(tree: uproot.TTree, backend: Literal['pd', 'ak'] = 'pd', downcast: bool = True) -> ROOTVariables:
    """Reads a ROOT TTree and returns a dictionary of Tensorflow `tf.RaggedTensor` or `tf.Tensor` objects. The keys are 
    the variable names and the values read from the TTree. Converting the TTree is done by a variable at a time. 

    Args:
        tree (uproot.TTree): ROOT TTree to be read.
        backend (str, optional): 'pd' or 'ak'. Backend to use for reading the TTree, 'pd' is faster but consumes more memory. Defaults to 'pd'.
        downcast (bool, optional): Downcast the output to `tf.float32`, `tf.int32` or `tf.uint32`. Defaults to True.

    Raises:
        ValueError: If backend is not 'pd' or 'ak'.

    Returns:
        ROOTVariables: Dictionary of Tensorflow `tf.RaggedTensor` or `tf.Tensor` objects. The keys are the variable names and the values read from the TTree.
    """

    if backend != 'pd' and backend != 'ak':
        raise ValueError(
            f'Backend {backend} not supported. Choose from pd (pandas) or ak (awkward).')
    variables = tree.keys()
    output = {}
    for var in variables:
        var_branch = tree[var].array(library="ak")
        if ak.num(ak.flatten(var_branch, axis=None), axis=0) == 0:
            continue
        if backend == 'ak':
            tensor = awkward_to_tensor(var_branch)
        elif backend == 'pd':
            var_branch = ak.to_dataframe(var_branch)
            if var_branch.empty:
                continue
            tensor = pandas_to_tensor(var_branch['values'])

        if downcast:
            if tensor.dtype == tf.float64:
                tensor = tf.cast(tensor, FLOAT_PRECISION)
            elif tensor.dtype == tf.int64:
                tensor = tf.cast(tensor, INT_PRECISION)
            elif tensor.dtype == tf.uint64:
                tensor = tf.cast(tensor, INT_PRECISION)

        output[var] = tensor
        logging.info(f'{var}: {output[var].shape} {output[var].dtype}')
    return output


@tf.function
def dict_to_stacked_tensor(data: Union[ROOTVariables, Tuple[ROOTVariables, ROOTVariables]], ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
    """Converts a `ROOTVariables` to a input for training a neural network, i.e. a tuple `(input, label, weight)`.
    The `input` is construsted by **stacking all the variables**  in `data` `ROOTVariables` dictionary into a single `tf.Tensor`.

    Optionally, the input data can be a tuple of two ROOTVariables. The output has the form `((input1, input2), label, weight)`.
    The `input2` is constructed by **stacking all the variables** in the second `ROOTVariables` dictionary into a single `tf.Tensor`.

    Args:
        data (ROOTVariables or tuple[ROOTVariables, ROOTVariables]): The input data.
        label (int): The label.
        weight (float, optional): The weight. Defaults to `None`.

    Returns:
        A tuple `(input, label, weight)` where `data` is a  `tf.Tensor` or a tuple `((input1, input2), label, weight)`
        in case `data` is a tuple of two ROOTVariables where `input1` and `input2` are `tf.Tensor`s.
    """

    if isinstance(data, tuple):
        output_data = []
        for data_input in data:
            output_data.append(tf.stack(
                [tf.cast(data_input[var], FLOAT_PRECISION) for var in data_input.keys()], axis=-1))
        return tuple(output_data)
    else:
        return tf.stack([tf.cast(data[var], FLOAT_PRECISION) for var in data.keys()], axis=-1)


def get_var_picker(target: Optional[str] = None,
                   weight: Optional[str] = None,
                   variables: Optional[List[str]] = None) -> Callable[[ROOTVariables],
                                                                      Union[Tuple[ROOTVariables, tf.RaggedTensor, tf.RaggedTensor], ROOTVariables, Tuple[ROOTVariables, tf.RaggedTensor]]]:
    @tf.function
    def pick_variables(sample: ROOTVariables) -> Union[Tuple[ROOTVariables, tf.RaggedTensor, tf.RaggedTensor], ROOTVariables, Tuple[ROOTVariables, tf.RaggedTensor]]:
        keys = sample.keys() if variables is None else variables
        new_sample = {var: Expression(var)(sample)
                      for var in keys if var not in [target, weight]}

        if target is None:
            return new_sample
        if weight is None:
            return new_sample, Expression(target)(sample)
        else:
            return new_sample, Expression(target)(sample), Expression(weight)(sample)
    return pick_variables


def root_file_iterator(filename: str,
                       treename: str,
                       step_size: Optional[int] = None,
                       entry_start: Optional[int] = None,
                       entry_stop: Optional[int] = None,
                       num_workers: int = 1,
                       downcast: bool = True,
                       manual_cast_int: List[str] = [],
                       manual_cast_float: List[str] = []) -> Iterator[Dict[str, Union[tf.Tensor, tf.RaggedTensor]]]:
    """A generator that yields batches of data from a ROOT file as dictionaries of tensors. 
    Tensors are created from the awkward arrays into normal tensors or ragged tensors, 
    optionaly with multiple raggged dimensions based on the structure of the awkward array.

    Args:
        filename (str): ROOT file name
        treename (str): name of the tree in the ROOT file
        step_size (Optional[int], optional): Size of the batch. If None, the whole tree is loaded. Defaults to None.
        entry_start (Optional[int], optional): Index of the first entry to load. If None, the whole tree is loaded. Defaults to None.
        entry_stop (Optional[int], optional): Index of the last entry to load. Defaults to None.
        num_workers (int, optional): Number of workers to use for parallel loading in `uproot`. Defaults to 1.
        downcast (bool, optional): Downcast the output to `tf.float32`, `tf.int32` or `tf.uint32`. Defaults to True.
        manual_cast_int (List[str], optional): List of variables to cast to `tf.int32`. Defaults to [].
        manual_cast_float (List[str], optional): List of variables to cast to `tf.float32`. Defaults to [].

    Yields:
        Dict[str, Union[tf.Tensor, tf.RaggedTensor]]: Dictionary of tensors with the same keys as the branches in the ROOT tree.
    """

    with uproot.open(filename, num_workers=num_workers) as file:
        tree = file[treename]
        variables = tree.keys()
        num_entries = tree.num_entries if entry_stop is None else entry_stop - entry_start
        processed_events = 0
        for i, batch in enumerate(tree.iterate(step_size=step_size, entry_start=entry_start, entry_stop=entry_stop)):

            processed_events += ak.num(batch[variables[0]], axis=0)

            logging.info(
                f"Processing chunk {i}: loaded {processed_events} out of {num_entries} events ({100*processed_events/num_entries if num_entries > 0 else 0:.1f}%).")

            output = {}
            for var in variables:
                tensor = awkward_to_tensor(batch[var])
                if downcast:
                    if tensor.dtype == tf.float64:
                        tensor = tf.cast(tensor, FLOAT_PRECISION)
                    elif tensor.dtype == tf.int64:
                        tensor = tf.cast(tensor, INT_PRECISION)
                    elif tensor.dtype == tf.uint64:
                        tensor = tf.cast(tensor, INT_PRECISION)

                if var in manual_cast_int:
                    tensor = tf.cast(tensor, INT_PRECISION)
                elif var in manual_cast_float:
                    tensor = tf.cast(tensor, FLOAT_PRECISION)

                output[var] = tensor
            yield output
            # yield {var: awkward_to_tensor(batch[var]) for var in variables}


@ray.remote
class BatchSaver:
    """A class that saves a batch of data to a folder as a tf.data.Dataset.
    This is a workaround for the memory leak in tf.data.Dataset.save().

    Args:
        folder (str): Folder to save the batch to.
        data (Dict[str, Union[tf.Tensor, tf.RaggedTensor]]): Dictionary of tensors to save.
    """

    def __init__(self, folder: str, data: Dict[str, Union[tf.Tensor, tf.RaggedTensor]]):
        self.folder = folder
        self.data = data

    def save(self):
        """Saves the batch to the folder.

        Returns:
            Tuple[Any, Any]: The element spec and cardinality of the saved dataset.
        """
        dataset = tf.data.Dataset.from_tensor_slices(self.data)
        dataset.save(self.folder, compression='GZIP')
        return (dataset.element_spec, dataset.cardinality())


def save_batch(folder: str, data: Dict[str, Union[tf.Tensor, tf.RaggedTensor]], n_parallel: Optional[int] = None) -> Tuple[Any, Any]:
    """Saves a batch of data to a folder as a tf.data.Dataset using ray actors.

    Args:
        folder (str): Folder to save the batch to.
        data (Dict[str, Union[tf.Tensor, tf.RaggedTensor]]): Dictionary of tensors to save.

    Returns:
        Tuple[Any, Any]: The element spec and cardinality of the saved dataset.
    """
    with ray.init(num_cpus=n_parallel):
        actors = [BatchSaver.remote(folder, data)]
        output = ray.get([actor.save.remote() for actor in actors])
        element_spec, cardinality = output[0]
    return element_spec, cardinality


def data_iterator_to_dataset(iterator: Iterator[Dict[str, Union[tf.Tensor, tf.RaggedTensor]]],
                             tmp_folder: str,
                             single_batch: bool = False,
                             n_parallel: Optional[int] = None) -> tf.data.Dataset:
    """Converts a data iterator to a tf.data.Dataset. The iterator consists of batches of data in a form of dictionaries of tensors.
    Each of the tensor has an extra dimension that corresponds to the batch size. The function saves each batch to a separate folder
    and then loads the batches as tf.data.Datasets. The datasets are then merged into one dataset using tf.data.Dataset.sample_from_datasets().
    This is a workaround for the memory leak in tf.data.Dataset.save() and to parallelize the loading of the batches.

    Args:
        iterator (Iterator[Dict[str, Union[tf.Tensor, tf.RaggedTensor]]]): Iterator of batches of data. Each batch is a dictionary of tensors, where each tensor has an extra dimension that corresponds to the batch size.
        tmp_folder (str): Folder to save the batches to.

    Returns:
        tf.data.Dataset: Dataset containing all the batches.
    """

    if single_batch:
        return tf.data.Dataset.from_tensor_slices(next(iterator))

    batch_folders = []
    element_specs = []
    cardinalities = []
    for i, batch in enumerate(iterator):
        # check if iterator is empty at the beginning of the loop
        batch_folder = os.path.join(tmp_folder, f"batch_{i}")
        os.makedirs(batch_folder, exist_ok=True)
        element_spec, cardinality = save_batch(batch_folder, batch, n_parallel)
        batch_folders.append(str(batch_folder))
        element_specs.append(element_spec)
        cardinalities.append(cardinality)

    dataset = tf.data.Dataset.sample_from_datasets(
        [tf.data.Dataset.load(batch_folder, compression='GZIP', element_spec=element_spec)
         for batch_folder, element_spec in zip(batch_folders, element_specs)],
        stop_on_empty_dataset=False,
    )
    dataset = dataset.apply(
        tf.data.experimental.assert_cardinality(sum(cardinalities)))
    return dataset


class JIDENNDataset:
    """The JIDENNDataset dataclass is a wrapper for a TensorFlow dataset that allows for easy loading and processing of dataset files 
    for jet identifiation using deep neural networks (**JIDENN**). The `tf.data.Dataset` is constructed from a `tf.data.Dataset` 
    consisting of `ROOTVariables` dictionaries. 

    The dataset can be loaded from a file using the `load_dataset` method or set manually using the `set_dataset` method.
    Both methods require the `element_spec` either in a pickled file in the case of loading, or as dictionary of `tf.TensorSpec` 
    or `tf.RaggedTensorSpec`  object in the case of setting the dataset manually.

    Example:
    Typical usage of the `JIDENNDataset` dataclass is as follows:
    ```python
    import tensorflow as tf
    from jidenn.config.config_subclasses import Variables
    from .utils.Cut import Cut

    @tf.function 
    def count_PFO(sample: ROOTVariables) -> ROOTVariables:
        sample = sample.copy()
        sample['jets_PFO_n'] = tf.reduce_sum(tf.ones_like(sample['jets_PFO_pt']))
        return sample

    @tf.function 
    def train_input(sample: ROOTVariables) -> ROOTVariables:
        output = {
            'N_PFO': sample['jets_PFO_n'],
            'pt': sample['jets_pt'],
            'width': sample['jets_Width'],
            'EMFrac': sample['jets_EMFrac'],
            'mu': sample['corrected_averageInteractionsPerCrossing[0]']
        }
        return output

    variables = ['corrected_averageInteractionsPerCrossing[0]', 'jets_pt', 'jets_Width', 'jets_EMFrac','jets_PFO_pt']

    jidenn_dataset = JIDENNDataset(variables=variables,
                                   target='jets_TruthLabelID',
                                   weight=None)
    jidenn_dataset = jidenn_dataset.load_dataset('path/to/dataset')

    jidenn_dataset = jidenn_dataset.create_variables(cut=Cut('jets_pt > 10_000'), map_dataset=count_PFO)
    jidenn_dataset = jidenn_dataset.resample_dataset(lambda data, label: tf.cast(tf.greater(label, 0), tf.int32), [0.5, 0.5])
    jidenn_dataset = jidenn_dataset.remap_labels(lambda data, label: tf.cast(tf.greater(label, 0), tf.int32))
    jidenn_dataset = jidenn_dataset.create_train_input(train_input)
    dataset = jidenn_dataset.get_prepared_dataset(batch_size=128, 
                                                  shuffle_buffer_size=1000, 
                                                  take=100_000,
                                                  assert_length=True)
    model.fit(dataset, epochs=10)
    ```
    """

    def __init__(self, dataset: tf.data.Dataset,
                 element_spec: Dict[str, Union[tf.TensorSpec, tf.RaggedTensorSpec]],
                 metadata: Optional[Dict[str, Any]] = None,
                 variables: Optional[List[str]] = None, target: Optional[str] = None, weight: Optional[str] = None, length: Optional[int] = None):
        """Initializes the JIDENNDataset object. """

        if dataset is not None and element_spec is None:
            logging.warning(
                "Element spec not set. Using the element spec of the dataset.")
            element_spec = dataset.element_spec
        self._dataset = dataset
        self._element_spec = element_spec
        self._metadata = metadata
        self._variables = variables
        self._target = target
        self._weight = weight
        self._length = length

    @property
    def dataset(self) -> Union[tf.data.Dataset, None]:
        """The `tf.data.Dataset` object or `None` if the dataset is not set yet."""
        return self._dataset

    @property
    def element_spec(self) -> Union[Dict[str, Union[tf.TensorSpec, tf.RaggedTensorSpec]], None]:
        """The `element_spec` of the dataset or `None` if the dataset is not set yet."""
        return self._element_spec

    @property
    def metadata(self) -> Union[Dict[str, Any], None]:
        """The metadata of the dataset or `None` if the dataset is not set yet.
        Dict of tensorflow `tf.Tensor` objects containing the metadata of the original ROOT file. The metadata is a histogram containing
        the num events, sum of weights, etc. of the ROOT file. The metadata is read from the ROOT file using the 
        `h_metadata` histogram. If the histogram is not present, the metadata is `None`.
        """
        return self._metadata

    @property
    def variables(self) -> Union[List[str], None]:
        """The variables of the dataset or `None` if the dataset is not set yet."""
        if self.dataset is None:
            raise ValueError('Dataset not loaded yet.')
        if self._variables is not None:
            return self._variables
        if not isinstance(self.element_spec, tuple):
            return list(self.element_spec.keys())
        elif isinstance(self.element_spec[0], tuple):
            return list(self.element_spec[0][0].keys()) + list(self.element_spec[0][1].keys())
        else:
            return list(self.element_spec[0].keys())

    @property
    def target(self) -> Union[str, None]:
        """The target of the dataset or `None` if the dataset is not set yet."""
        return self._target

    @property
    def weight(self) -> Union[str, None]:
        """The weight of the dataset or `None` if the dataset is not set yet."""
        return self._weight

    @property
    def length(self) -> Union[int, None]:
        """The length of the dataset or `None` if the dataset is not set yet."""
        return self._length

    @staticmethod
    def load(path: Union[str, List[str]],
             element_spec_path: Optional[str] = None,
             metadata_path: Optional[str] = None,
             shuffle_reading: bool = False) -> JIDENNDataset:
        """Loads a dataset from a file. The dataset is stored in the `tf.data.Dataset` format.
        The assumed dataset elements are `ROOTVariables` dictionaries or a tuple of `ROOTVariables`, `label` and `weight`.

        Args:
            path (str): The path to the dataset directory.
            element_spec_path (str, optional): The path to the `element_spec` file. Defaults to `None`. 
                If `None`, the `element_spec` is loaded from the `element_spec` file inside the dataset directory.

        Raises:
            ValueError: If the `element_spec` is not a dictionary or a tuple whose first element is a dictionary.

        Returns:
            JIDENNDataset: The JIDENNDataset object with set dataset and `element_spec`.
        """
        if isinstance(path, list) and element_spec_path is None and metadata_path is None:
            return JIDENNDataset.load_multiple(path, element_spec_path, metadata_path)
        elif isinstance(path, list):
            raise ValueError(
                'Multiple datasets can only be loaded if `element_spec_path` and `metadata_path` are set. Otherwise use `load_multiple`.')

        element_spec_path = os.path.join(
            path, 'element_spec.pkl') if element_spec_path is None else element_spec_path
        with open(element_spec_path, 'rb') as f:
            element_spec = pickle.load(f)

        metadata_path = os.path.join(
            path, 'metadata.pkl') if metadata_path is None else metadata_path
        try:
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
        except FileNotFoundError:
            logging.warning(f'Metadata file `{metadata_path}` not found.')
            metadata = None

        if isinstance(element_spec, dict):
            variables = list(element_spec.keys())

        elif isinstance(element_spec[0], dict):
            variables = list(element_spec[0].keys())

        else:
            raise ValueError('Element spec is not a dictionary.')

        @tf.function
        def shuffle_reading_fn(datasets):
            datasets = datasets.shuffle(512)
            return datasets.interleave(lambda x: x, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = tf.data.Dataset.load(
            path, compression='GZIP', element_spec=element_spec, reader_func=shuffle_reading_fn if shuffle_reading else None)

        return JIDENNDataset(dataset=dataset, element_spec=element_spec, metadata=metadata, variables=variables, length=dataset.cardinality().numpy())

    @staticmethod
    def load_multiple(files: List[str],
                      file_labels: Optional[List[Any]] = None,
                      dataset_mapper: Optional[Callable[[
                          tf.data.Dataset], tf.data.Dataset]] = None,
                      element_spec_paths: Optional[List[str]] = None,
                      stop_on_empty_dataset: bool = False,
                      weights: Optional[List[float]] = None,
                      mode: Literal['concatenate',
                                    'interleave',
                                    'sample'] = 'interleave',
                      metadata_combiner: Optional[Callable[[List[Dict[str, Any]]], Dict[str, Any]]] = lambda metas: {
                          key: tf.reduce_sum(tf.stack([m[key] for m in metas], axis=0), axis=0) for key in metas[0]},
                      metadata_paths: Optional[List[str]] = None) -> JIDENNDataset:

        element_spec_paths = [
            None] * len(files) if element_spec_paths is None else element_spec_paths
        metadata_paths = [None] * \
            len(files) if metadata_paths is None else metadata_paths
        file_labels = [None] * \
            len(files) if file_labels is None else file_labels

        dss = []
        for file, file_label, element_spec_path, metadata_path in zip(files, file_labels, element_spec_paths, metadata_paths):
            ds = JIDENNDataset.load(file, element_spec_path, metadata_path)
            ds = ds.apply(lambda x: dataset_mapper(x, file_label)
                          ) if dataset_mapper is not None else ds
            dss.append(ds)

        return JIDENNDataset.combine(dss, stop_on_empty_dataset=stop_on_empty_dataset, mode=mode, weights=weights, metadata_combiner=metadata_combiner)

    @staticmethod
    def load_parallel(files: List[str],
                      file_labels: Optional[List[Any]] = None,
                      element_spec_paths: Optional[List[str]] = None,
                      take: Optional[int] = None,
                      dataset_mapper: Optional[Callable[[
                          tf.data.Dataset], tf.data.Dataset]] = None,
                      num_parallel_calls: int = tf.data.AUTOTUNE,
                      metadata_combiner: Optional[Callable[[List[Dict[str, Any]]], Dict[str, Any]]] = lambda metas: {
                          key: tf.reduce_sum(tf.stack([m[key] for m in metas], axis=0), axis=0) for key in metas[0]},
                      metadata_paths: Optional[Union[List[str], str]] = None) -> JIDENNDataset:

        if file_labels is not None and len(file_labels) != len(files):
            raise ValueError(
                f'Number of file labels ({len(file_labels)}) must be equal to the number of files ({len(files)}).')

        # Load metadata
        metadata_paths = [metadata_paths] if isinstance(
            metadata_paths, str) else metadata_paths
        m_paths = [os.path.join(file, 'metadata.pkl')
                   for file in files] if metadata_paths is None else metadata_paths
        metadatas = []
        for metadata_path in m_paths:
            try:
                with open(metadata_path, 'rb') as f:
                    metadatas.append(pickle.load(f))
            except FileNotFoundError:
                logging.warning(f'Metadata file `{metadata_path}` not found.')
                metadatas.append(None)

        if not all_equal([metadata.keys() for metadata in metadatas]):
            raise ValueError(
                f'All datasets must have the same metadata keys: {[metadata.keys() for metadata in metadatas]}')
        elif metadata_paths is not None and len(metadata_paths) == 1:
            logging.info('Only one metadata file found.')
            metadata = metadatas[0]
        elif metadata_combiner is not None:
            logging.info('Combining metadata.')
            metadata = metadata_combiner(metadatas)
        else:
            logging.info('Not using metadata.')
            metadata = None

        # Load element spec
        es_paths = [os.path.join(file, 'element_spec.pkl')
                    for file in files] if element_spec_paths is None else element_spec_paths
        element_specs = []
        for element_spec_path in es_paths:
            with open(element_spec_path, 'rb') as f:
                element_specs.append(pickle.load(f))

        if not all_equal(element_specs):
            raise ValueError('All datasets must have the same element spec.')

        if file_labels is None:
            @tf.function
            def _interleaver(file):
                dataset = tf.data.Dataset.load(
                    file, compression='GZIP', element_spec=element_specs[0])
                if take is not None:
                    dataset = dataset.take(take)
                if dataset_mapper is not None:
                    dataset = dataset_mapper(dataset)
                return dataset
        else:
            @tf.function
            def _interleaver(file, label):
                dataset = tf.data.Dataset.load(
                    file, compression='GZIP', element_spec=element_specs[0])
                if take is not None:
                    dataset = dataset.take(take)
                if dataset_mapper is not None:
                    dataset = dataset_mapper(dataset, label)
                return dataset

        dataset = tf.data.Dataset.from_tensor_slices(files)
        if file_labels is not None:
            label_dataset = tf.data.Dataset.from_tensor_slices(
                tf.constant(file_labels))
            dataset = tf.data.Dataset.zip((dataset, label_dataset))
        dataset = dataset.interleave(
            _interleaver, num_parallel_calls=num_parallel_calls)
        element_spec = dataset.element_spec

        return JIDENNDataset(dataset=dataset, element_spec=element_spec, metadata=metadata, length=None)

    @staticmethod
    def from_tf_dataset(dataset: tf.data.Dataset, metadata: Optional[Dict[str, Any]] = None):
        return JIDENNDataset(dataset=dataset,
                             metadata=metadata,
                             element_spec=dataset.element_spec,
                             length=dataset.cardinality().numpy())

    @staticmethod
    def combine(datasets: List[JIDENNDataset],
                mode: Literal['concatenate', 'interleave',
                              'sample'] = 'concatenate',
                stop_on_empty_dataset: bool = False,
                weights: Optional[List[float]] = None,
                metadata_combiner: Optional[Callable[[List[Dict[str, Any]]], Dict[str, Any]]] = lambda metas: {key: tf.reduce_sum(tf.stack(metas[key], axis=0), axis=0) for key in metas[0]}) -> JIDENNDataset:
        """Combines multiple datasets into one dataset. The samples are interleaved and the weights are used to sample from the datasets.

        Args:
            datasets (List[JIDENNDataset]): List of datasets to combined. All `JIDENNDataset.dataset`s must be set and have the same `element_spec`.
            mode (str, optional): 'concatenate' or 'interleave'. The mode to combine the datasets. Defaults to 'concatenate'.
            dataset_weights (List[float]): List of weights for each dataset. The weights are used to sample from the datasets.
            metadata_combiner (Optional[Callable[[List[Dict[str, Any]]], Dict[str, Any]]], optional): Function that combines the metadata of the datasets. 
                The function takes a list of metadata dictionaries and returns a single metadata dictionary.
                Defaults to lambda metas: {key: tf.reduce_sum(tf.stack(metas, axis=0), axis=0) for key in metas[0]}.

        Returns:
            JIDENNDataset: Combined `JIDENNDataset` object.
        """
        element_specs = [dataset.element_spec for dataset in datasets]
        variables = [dataset.variables for dataset in datasets]
        targets = [dataset.target for dataset in datasets]
        weight = [dataset.weight for dataset in datasets]

        if not all_equal(element_specs):
            raise ValueError('All datasets must have the same element spec.')
        if not all_equal(variables):
            raise ValueError('All datasets must have the same variables.')
        if not all_equal(targets):
            raise ValueError('All datasets must have the same target.')
        if not all_equal(weight):
            raise ValueError('All datasets must have the same weight.')
        if sum_metadata and not all_equal([dataset.metadata.keys() for dataset in datasets]):
            raise ValueError('All datasets must have the same metadata.')

        if mode == 'sample':
            dataset = tf.data.Dataset.sample_from_datasets(
                [dataset.dataset for dataset in datasets], stop_on_empty_dataset=stop_on_empty_dataset, weights=weights)
        elif mode == 'concatenate':
            logging.warning('Weights are ignored in concatenate mode.') if weights is not None else None
            dataset = datasets[0].dataset
            for ji_ds in datasets[1:]:
                dataset = dataset.concatenate(ji_ds.dataset)
        elif mode == 'interleave':
            dataset = tf.data.Dataset.from_tensor_slices([dataset.dataset for dataset in datasets]).interleave(
                lambda x: x, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            raise ValueError(
                f'Mode {mode} not supported. Choose from concatenate or interleave.')

        if datasets[0].metadata is not None and metadata_combiner is not None:
            metadata = metadata_combiner([ds.metadata for ds in datasets])
        else:
            metadata = None

        lengths = [ds.length for ds in datasets]
        if all(length is not None for length in lengths):
            length = sum(lengths)
        else:
            length = None
        return JIDENNDataset(dataset=dataset, metadata=metadata,
                             element_spec=element_specs[0],
                             variables=variables[0],
                             target=targets[0],
                             weight=weight[0],
                             length=length)

    @staticmethod
    def from_root_file(filename: str,
                       tree_name: str = 'NOMINAL',
                       metadata_hist: Optional[str] = 'h_metadata',
                       backend: Literal['pd', 'ak'] = 'pd') -> JIDENNDataset:
        """Reads a ROOT file and returns a `JIDENNDataset` object. 

        Args:
            filename (str): Path to the ROOT file.
            tree_name (str, optional): Name of the TTree in the ROOT file. Defaults to 'NOMINAL'.
            metadata_hist (str, optional): Name of the histogram containing the metadata. Defaults to 'h_metadata'. Could be `None`.
            backend (str, optional): 'pd' or 'ak'. Backend to use for reading the TTree, 'pd' is faster but consumes more memory. Defaults to 'pd'.

        Returns:
            JIDENNDataset: `JIDENNDataset` object.
        """
        with uproot.open(filename, object_cache=None, array_cache=None) as file:
            tree = file[tree_name]
            logging.info(f"Loading ROOT file {filename}")
            sample = read_ttree(tree, backend=backend)

            if metadata_hist is not None:
                logging.info("Getting metadata")
                labels = file[metadata_hist].member('fXaxis').labels()
                values = file[metadata_hist].values()
                values = tf.constant(values)
                metadata = dict(zip(labels, values))
                logging.info(f"Metadata: {metadata}")
            else:
                metadata = None

            logging.info(f'Done loading file:{filename}')
            dataset = tf.data.Dataset.from_tensor_slices(sample)

        variables = list(dataset.element_spec.keys())
        element_spec = dataset.element_spec
        target = None
        weight = None
        return JIDENNDataset(dataset=dataset, element_spec=element_spec,
                             metadata=metadata, variables=variables, target=target,
                             weight=weight, length=dataset.cardinality().numpy())

    @staticmethod
    def from_root_file_batched(filename: str,
                               step_size: int = 10,
                               entry_start: Optional[int] = None,
                               entry_stop: Optional[int] = None,
                               tmp_folder: str = 'tmp',
                               n_parallel: int = 1,
                               tree_name: str = 'NOMINAL',
                               metadata_hist: Optional[str] = 'h_metadata',
                               manual_cast_int: List[str] = [],
                               manual_cast_float: List[str] = []) -> JIDENNDataset:

        if metadata_hist is not None:
            with uproot.open(filename) as file:
                logging.info("Getting metadata")
                labels = file[metadata_hist].member('fXaxis').labels()
                values = file[metadata_hist].values()
                values = tf.constant(values)
                metadata = dict(zip(labels, values))
                logging.info(f"Metadata: {metadata}")
        else:
            metadata = None

        iterator = root_file_iterator(filename, tree_name, step_size=step_size,
                                      entry_start=entry_start, entry_stop=entry_stop, num_workers=n_parallel,
                                      manual_cast_int=manual_cast_int, manual_cast_float=manual_cast_float)

        single_batch = entry_stop - \
            entry_start <= step_size if entry_start is not None and entry_stop is not None else False

        dataset = data_iterator_to_dataset(
            iterator, tmp_folder, single_batch=single_batch, n_parallel=n_parallel)

        variables = list(dataset.element_spec.keys())
        element_spec = dataset.element_spec
        target = None
        weight = None
        return JIDENNDataset(dataset=dataset, element_spec=element_spec,
                             metadata=metadata, variables=variables, target=target,
                             weight=weight, length=dataset.cardinality().numpy())

    def set_variables_target_weight(self,
                                    variables: Optional[List[str]] = None,
                                    target: Optional[str] = None,
                                    weight: Optional[str] = None) -> JIDENNDataset:
        """Sets the `variables`, `target` and `weight` of the dataset."""
        if self.dataset is None:
            raise ValueError('Dataset not loaded yet.')
        dataset = self.dataset.map(get_var_picker(
            target=target, weight=weight, variables=variables))
        new_vars = list(dataset.element_spec[0].keys())
        return JIDENNDataset(dataset=dataset,
                             metadata=self.metadata,
                             element_spec=dataset.element_spec,
                             variables=new_vars,
                             target=target,
                             weight=weight, length=self.length)

    def save(self, file: str, num_shards: Optional[int] = None, **kwargs) -> None:
        """Saves the dataset to a file. The dataset is stored in the `tf.data.Dataset` format.
        The `element_spec` is stored in the `element_spec` file inside the dataset directory.
        Tensorflow saves the `element_spec.pb` automatically, but manual save is required 
        for further processing of the dataset.  Ternsorflow file has the `.pb` extension.

        Args:
            file (str): The path to the dataset directory.
            num_shards (int, optional): The number of shards to split the dataset into. Defaults to `None`. The sharding is done uniformly into `num_shards` files.

        Raises:
            ValueError: If the dataset is not loaded yet.

        Returns:
            None
        """

        if self.dataset is None:
            raise ValueError('Dataset not loaded yet.')

        @tf.function
        def random_shards(_) -> tf.Tensor:
            return tf.random.uniform(shape=[], minval=0, maxval=num_shards, dtype=tf.int64)

        self.dataset.save(file, compression='GZIP',
                          shard_func=random_shards if num_shards is not None else None, **kwargs)

        with open(os.path.join(file, 'element_spec.pkl'), 'wb') as f:
            pickle.dump(self.dataset.element_spec, f)

        if self.metadata is not None:
            with open(os.path.join(file, 'metadata.pkl'), 'wb') as f:
                pickle.dump(self.metadata, f)

    def remap_labels(self, label_mapping: Callable[[int], int]) -> JIDENNDataset:
        """Remaps the labels in the dataset using the `label_mapping` function.
        Should be used after the `create_variables` method. 

        Args:
            label_mapping (Callable[[int], int]): The function that maps the labels.

        Raises:
            ValueError: If the dataset is not loaded yet.
            ValueError: If the `target` is not set.

        Returns:
            JIDENNDataset: The JIDENNDataset object where the `label` is remapped.
        """
        if self.dataset is None:
            raise ValueError('Dataset not loaded yet.')
        if self.target is None:
            raise ValueError('Target not set yet.')
        if not isinstance(self.element_spec, tuple):
            raise ValueError(
                'Dataset element spec must be a tuple. Use `create_variables` method separate target and weight as tuple.')

        if self.weight is not None:
            @tf.function
            def remap_label(x, y, w):
                return x, label_mapping(y), w
        else:
            @tf.function
            def remap_label(x, y):
                return x, label_mapping(y)

        dataset = self.dataset.map(remap_label)
        return JIDENNDataset(dataset=dataset, element_spec=dataset.element_spec,
                             metadata=self.metadata, variables=self.variables,
                             target=self.target, weight=self.weight, length=self.length)

    def remap_data(self, func: Callable[[ROOTVariables], Union[ROOTVariables, Tuple[ROOTVariables, ROOTVariables]]]) -> JIDENNDataset:
        """Remaps the data of the tf.data.Dataset using the `func` function. The function must take a `ROOTVariables` object and return a `ROOTVariables` object.
        The output of the function is of the form Dict[str, tf.Tensor] or Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]] (optionally also tf.RaggedTensor).


        Args:
            func (Callable[[ROOTVariables], Union[ROOTVariables, Tuple[ROOTVariables, ROOTVariables]]]): Function to apply to the data of the dataset.

        Raises:
            ValueError: If the dataset is not loaded yet.

        Returns:
            JIDENNDataset: The JIDENNDataset object  with signature `((ROOTVariables, ROOTVariables), ...)` or `(ROOTVariables, ...)`.
        """
        if self.dataset is None:
            raise ValueError('Dataset not loaded yet.')

        if self.weight is not None:
            @tf.function
            def input_wrapper(data, label, w):
                return func(data), label, w
        elif self.target is not None:
            @tf.function
            def input_wrapper(data, label):
                return func(data), label
        else:
            @tf.function
            def input_wrapper(*data):
                return func(*data)
        dataset = self.dataset.map(input_wrapper)
        return JIDENNDataset(dataset=dataset, element_spec=dataset.element_spec,
                             metadata=self.metadata,
                             target=self.target, weight=self.weight, length=self.length)

    def resample_dataset(self, resampling_func: Callable[[ROOTVariables, Any], int], target_dist: List[float]):
        """Resamples the dataset using the `resampling_func` function. The function computes the bin index for each sample in the dataset. 
        The dataset is then resampled to match the `target_dist` distribution. Be careful that this may **slow down the training process**,
        if the target distribution is very different from the original one as the dataset is resampled on the fly and is waiting 
        for the appropriate sample to be drawn.

        Args:
            resampling_func (Callable[[ROOTVariables, Any], int]): Function that bins the data. It must return an integer between 0 and `len(target_dist) - 1`.
            target_dist (List[float]): The target distribution of the resampled dataset.

        Raises:
            ValueError: If the dataset is not loaded yet.

        Returns:
            JIDENNDataset: The JIDENNDataset object where the dataset is resampled.
        """
        if self.dataset is None:
            raise ValueError('Dataset not loaded yet.')

        @tf.function
        def _data_only(x, data):
            return data
        dataset = self.dataset.rejection_resample(
            resampling_func, target_dist=target_dist).map(_data_only)
        return JIDENNDataset(dataset=dataset, element_spec=dataset.element_spec,
                             metadata=self.metadata, variables=self.variables,
                             target=self.target, weight=self.weight, length=None)

    def apply(self, func: Callable[[tf.data.Dataset], tf.data.Dataset], preserves_length: bool = False) -> JIDENNDataset:
        """Applies a function to the dataset.

        Args:
            func (Callable[[tf.data.Dataset], tf.data.Dataset]): Function to apply to the dataset.
            preserves_length (bool, optional): If `True`, the length of the dataset is preserved. Defaults to False.

        Returns:
            JIDENNDataset: The JIDENNDataset object with the dataset modified by the function.

        """
        if self.dataset is None:
            raise ValueError('Dataset not loaded yet.')
        dataset = func(self.dataset)
        return JIDENNDataset(dataset=dataset, element_spec=dataset.element_spec,
                             metadata=self.metadata, variables=self.variables,
                             target=self.target, weight=self.weight, length=self.length if preserves_length else None)

    def filter(self, filter: Callable[[ROOTVariables], bool]) -> JIDENNDataset:
        """Filters the dataset using the `filter` function. 

        Args:
            filter (Callable[[ROOTVariables], bool]): Function to apply to the data.

        Raises:
            ValueError: If the dataset is not loaded yet.

        Returns:
            JIDENNDataset: The JIDENNDataset object with the dataset filtered.
        """
        if self.dataset is None:
            raise ValueError('Dataset not loaded yet.')
        dataset = self.dataset.filter(filter)
        return JIDENNDataset(dataset=dataset, element_spec=dataset.element_spec,
                             metadata=self.metadata, variables=self.variables,
                             target=self.target, weight=self.weight, length=None)

    def get_prepared_dataset(self,
                             batch_size: Optional[int] = None,
                             assert_length: bool = False,
                             shuffle_buffer_size: Optional[int] = None,
                             ragged: bool = True,
                             take: Optional[int] = None,) -> tf.data.Dataset:
        """Returns a prepared dataset for training. The dataset is prepared by stacking the arrays in the `ROOTVariables` in the dataset using `dict_to_stacked_array`.
        The dataset is also batched, shuffled, shortend (using `take`) and mapped using the `map_func` function. The function is applied before the input is stacked.

        **Train input must be created with `JIDENNDataset.create_train_input` before calling this method.**  

        The assertion allows displaying the estimated epoch time during training. The assertion is only performed if `take` is set.

        Args:
            batch_size (int): Batch size of the dataset.
            assert_length (bool, optional): If `True`, the dataset is asserted to have the `take` length. It is only used if 'take' is set. Defaults to False.
            shuffle_buffer_size (int, optional): Size of the shuffle buffer. If `None`, the dataset is not shuffled. Defaults to None.
            take (int, optional): Number of elements to take from the dataset. If `None`, the dataset is not taken. Defaults to None.
            map_func (Callable[[Union[ROOTVariables, Tuple[ROOTVariables, ROOTVariables]], Any], Tuple[Union[ROOTVariables, Tuple[ROOTVariables, ROOTVariables]], Any]], optional): Function to apply to the dataset. Defaults to None.

        Raises:
            ValueError: If the dataset is not loaded yet.

        Returns:
            tf.data.Dataset: The prepared dataset.
        """

        if self.dataset is None:
            raise ValueError('Dataset not loaded yet.')

        self = self.remap_data(dict_to_stacked_tensor)
        dataset = self.dataset.shuffle(
            shuffle_buffer_size) if shuffle_buffer_size is not None else self.dataset
        dataset = dataset.take(take) if take is not None else dataset
        if assert_length and take is not None:
            dataset = dataset.apply(tf.data.experimental.assert_cardinality(
                take))
        elif assert_length and self.length is not None:
            dataset = dataset.apply(tf.data.experimental.assert_cardinality(
                self.length))
        if batch_size is not None and ragged:
            try:
                dataset = dataset.ragged_batch(batch_size)
            except AttributeError:
                dataset = dataset.apply(
                    tf.data.experimental.dense_to_ragged_batch(batch_size))

        elif batch_size is not None:
            dataset = dataset.batch(batch_size)

        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def take(self, n: int) -> JIDENNDataset:
        return self.apply(lambda dataset: dataset.take(n))

    def to_pandas(self, variables: Optional[List[str]] = None) -> pd.DataFrame:
        """Converts the dataset to a pandas DataFrame. The dataset must be loaded before calling this function.
        The function uses `tensorflow_datasets.as_dataframe` to convert the dataset to a pandas DataFrame, so 
        the `tensorflow_datasets` package must be installed.

        Be careful that this function may take a **long time to run**, depending on the size of the dataset.
        Consider taking only a subset of the dataset before converting it to a pandas DataFrame.
        ```python
        jidenn_dataset = JIDENNDataset(...) 
        ...
        jidenn_dataset = jidenn_dataset.apply(lambda dataset: dataset.take(1_000))
        df = jidenn_dataset.to_pandas()
        ```

        If the dataset contains nested tuples consider using `jidenn.data.data_info.explode_nested_variables` 
        on the tuple columns of the convereted dataframe.

        Args:
            variables (Optional[List[str]], optional): List of variables to convert to a pandas DataFrame. If `None`, all variables are converted. Defaults to `None`.

        Raises:
            ImportError: If `tensorflow_datasets` is not installed.
            ValueError: If the dataset is not loaded yet.

        Returns:
            pd.DataFrame: The `tf.data.Dataset` converted to a pandas `pd.DataFrame`.
        """

        if self.dataset is None:
            raise ValueError('Dataset not loaded yet.')

        if isinstance(self._element_spec, tuple) and variables is None:
            @tf.function
            def tuple_to_dict(data, label, weight=None):
                if isinstance(data, tuple):
                    new_data = {}
                    for dat in data:
                        new_data.update({**dat})
                else:
                    new_data = data
                weight = weight if weight is not None else tf.ones_like(label)
                data = {**new_data, 'label': label, 'weight': weight}
                return data

        elif isinstance(self._element_spec, tuple) and variables is not None:
            @tf.function
            def tuple_to_dict(data, label, weight=None):
                if isinstance(data, tuple):
                    new_data = {}
                    for dat in data:
                        new_data.update({**dat})
                else:
                    new_data = data
                weight = weight if weight is not None else tf.ones_like(label)
                data = {**new_data, 'label': label, 'weight': weight}
                return {k: data[k] for k in variables + ['label', 'weight']}

        elif isinstance(self._element_spec, dict) and variables is not None:
            @tf.function
            def tuple_to_dict(data):
                return {k: data[k] for k in variables}

        elif isinstance(self._element_spec, dict) and variables is None:
            @tf.function
            def tuple_to_dict(data):
                return data

        else:
            raise ValueError('The dataset must be a tuple or a dict.')

        dataset = self.dataset.map(tuple_to_dict)
        try:
            import tensorflow_datasets as tfds
            df = tfds.as_dataframe(dataset)
        except ImportError:
            logging.warning(
                'tensorflow_datasets is not installed. Using numpy instead.')
            df = pd.DataFrame(list(dataset.as_numpy_iterator()))
        df = df.rename(lambda x: x.replace('/', '.'), axis='columns')
        return df

    def to_numpy(self, variables: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Converts the dataset to a numpy arrays. The dataset must be loaded before calling this function.
        This function is **memory intensive** and may cause the program to crash if the dataset is too large.
        It must be called before creating train variables with `JIDENNDataset.create_variables`
        """
        variables = self.variables if variables is None else variables
        np_dataset = {}
        for var in variables:
            try:
                np_dataset[var] = np.asarray(
                    list(self.dataset.map(lambda x: x[var]).as_numpy_iterator()))
            except ValueError:
                logging.warning(
                    f'Variable {var} is a ragged tensor. Converting to a ragged tensor.')
                np_dataset[var] = tf.ragged.constant(
                    list(self.dataset.map(lambda x: x[var]).as_numpy_iterator()))
        return np_dataset

    def plot_data_distributions(self,
                                folder: str,
                                variables: Optional[List[str]] = None,
                                hue_variable: Optional[str] = None,
                                named_labels: Optional[Dict[int, str]] = None,
                                bins: int = 70,
                                xlabel_mapper: Optional[Dict[str, str]] = None) -> None:
        """Plots the data distributions of the dataset. The dataset must be loaded before calling this function.
        The function uses `jidenn.evaluation.plotter.plot_data_distributions` to plot the data distributions.

        Args:
            folder (str): The path to the directory where the plots are saved.
            variables (Optional[List[str]], optional): List of variables to plot. If `None`, all variables are plotted. Defaults to `None`.
            named_labels (Dict[int, str], optional): Dictionary mapping truth values to custom labels.
                If not provided, the truth values will be used as labels. 

        Raises:
            ValueError: If the dataset is not loaded yet.

        Returns:
            None
        """
        if self.dataset is None:
            raise ValueError('Dataset not loaded yet.')
        if not isinstance(self.element_spec, tuple):
            variables = list(self.element_spec.keys())
        elif isinstance(self.element_spec[0], tuple):
            variables = []
            for i in range(len(self.element_spec[0])):
                variables += list(self.element_spec[0][i].keys())
        else:
            variables = list(self.element_spec[0].keys())

        df = self.to_pandas(variables)
        plot_data_distributions(df, folder=folder, named_labels=named_labels, weight_variable='weight' if 'weight' in df.columns else None,
                                xlabel_mapper=xlabel_mapper, hue_variable=hue_variable, bins=bins)

    def plot_single_variable(self,
                             variable: str,
                             save_path: str,
                             hue_variable: Optional[str] = None,
                             weight_variable: Optional[str] = None,
                             **kwargs):

        convert_variables = [
            variable, hue_variable] if hue_variable is not None else [variable]
        if weight_variable is None:
            convert_variables += [self.weight] if self.weight is not None else []
        else:
            convert_variables += [weight_variable]
        df = self.to_pandas(convert_variables)
        plot_single_dist(df=df, variable=variable, save_path=save_path,
                         hue_var=hue_variable, weight_var=weight_variable, **kwargs)

    def split_train_dev_test(self,
                             train_fraction: float,
                             dev_fraction: float,
                             test_fraction: float,
                             backend: Literal['coin', 'cut'] = 'cut') -> Tuple[JIDENNDataset, JIDENNDataset, JIDENNDataset]:
        """Split a dataset into train, dev and test sets. The fractions must sum to 1.0 and only 2 decimal places 
        are taken into account. The cardinality of the returned datasets might not be known as the spliting is done
        by random filtering.

        Args:
            dataset (tf.data.Dataset): The dataset to split
            train_fraction (float): The fraction of the dataset to use for training dataset.
            dev_fraction (float): The fraction of the dataset to use for development dataset.
            test_fraction (float): The fraction of the dataset to use for testing dataset.

        Returns:
            Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: The train, dev and test datasets.
        """

        assert train_fraction + dev_fraction + \
            test_fraction == 1., "Fractions must sum to 1.0 and can only have 2 decimal places."

        dataset_size = self.length
        if dataset_size is not None and backend == 'cut':
            train_size = int(dataset_size * train_fraction)
            dev_size = int(dataset_size * dev_fraction)
            test_size = dataset_size - train_size - dev_size

            # Split the shuffled dataset into train, dev, and test datasets
            train = self.apply(lambda x: x.take(train_size))
            dev = self.apply(lambda x: x.skip(train_size).take(dev_size))
            test = self.apply(lambda x: x.skip(
                train_size + dev_size).take(test_size))
            return train, dev, test
        elif backend == 'cut':
            raise ValueError(
                'Cardinality of the dataset is not known. Use `backend=coin`.')

        @tf.function
        def random_number(sample: ROOTVariables) -> tf.Tensor:
            return sample, tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)

        @tf.function
        def train_filter(sample: ROOTVariables, random_number: tf.Tensor) -> tf.Tensor:
            return tf.greater(random_number, 0) and tf.less_equal(random_number, train_fraction)

        @tf.function
        def dev_filter(sample: ROOTVariables, random_number: tf.Tensor) -> tf.Tensor:
            return tf.greater(random_number, train_fraction) and tf.less_equal(
                random_number, train_fraction + dev_fraction)

        @tf.function
        def test_filter(sample: ROOTVariables, random_number: tf.Tensor) -> tf.Tensor:
            return tf.greater(random_number, train_fraction +
                              dev_fraction) and tf.less_equal(random_number, train_fraction + dev_fraction + test_fraction)

        @tf.function
        def delete_random_number(sample: ROOTVariables, random_number: tf.Tensor) -> ROOTVariables:
            return sample

        randomized_dataset = self.remap_data(random_number)

        train = randomized_dataset.filter(train_filter)
        train = train.remap_data(delete_random_number)
        dev = randomized_dataset.filter(
            dev_filter).remap_data(delete_random_number)
        test = randomized_dataset.filter(
            test_filter).remap_data(delete_random_number)
        return train, dev, test

    def histogram(self, binning: Union[List[Binning], Binning], weight_var:Optional[str] = None) -> Dict[str, np.ndarray]:
        if isinstance(binning, list):
            return histogram_multiple(self.dataset, binning, weight_var)
        else:
            return histogram(self.dataset, binning, weight_var)
        