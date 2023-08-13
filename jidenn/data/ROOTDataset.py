"""
Module for reading ROOT files and converting them to Tensorflow `tf.RaggedTensor` or `tf.Tensor` objects. 
The module contains the `ROOTDataset` class which is a wrapper of `tf.data.Dataset`.
It's main purpose is to read ROOT files and convert them to Tensorflow `tf.RaggedTensor` or `tf.Tensor` objects,
and to a `tf.data.Dataset` object afterwards. It relies on the `uproot` package.

Two ooptinal backends are available for converting ROOT files to Tensorflow objects: `pandas` and `awkward`.
"""
from __future__ import annotations
import tensorflow as tf
from typing import Callable, List, Union, Dict, Optional, Literal
import pickle
import os
import uproot
import logging
import pandas as pd
import awkward as ak


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
                tensor = tf.cast(tensor, tf.float32)
            elif tensor.dtype == tf.int64:
                tensor = tf.cast(tensor, tf.int32)
            elif tensor.dtype == tf.uint64:
                tensor = tf.cast(tensor, tf.uint32)

        output[var] = tensor
        logging.info(f'{var}: {output[var].shape} {output[var].dtype}')
    return output


class ROOTDataset:
    """Class to read a ROOT file and return a `tf.data.Dataset` object. The dataset contains a dictionary of Tensorflow
    `tf.RaggedTensor` or `tf.Tensor` objects. The keys are the variable names and the values read from the TTree.
    The `.root` files are read using `uproot` and the `tf.data.Dataset` is created using `tf.data.Dataset.from_tensor_slices`. 

    The ROOT file is read by a variable at a time, so the memory consumption may be high for large files. More precisely,
    the proces of creating the `tf.data.Dataset` from a dictionary of `tf.RaggedTensor` or `tf.Tensor` objects 
    **consumes a lot of memory**. This is done as a trade-off for higher conversion speed. 

    Example:
    ```python
    import tensorflow as tf

    root_file = 'path/to/file.root'
    save_path = 'path/to/save/dataset'
    root_dataset = ROOTDataset.from_root_file(root_file)
    root_dataset.save(save_path)
    ...
    root_dataset = ROOTDataset.load(save_path)
    dataset = root_dataset.dataset
    # Use as a training dataset 
    ```


    The initialization is only a convenience method. The `ROOTDataset.from_root_file` or `ROOTDataset.from_root_files`
    methods should be used for creating a `ROOTDataset` object instead.
    Args:
        dataset (tf.data.Dataset): Tensorflow `tf.data.Dataset` object.
        variables (list[str]): List of variable names.


    """

    def __init__(self, dataset: tf.data.Dataset, metadata: Optional[tf.Tensor] = None):
        self._dataset = dataset
        self._metadata = metadata

    @property
    def variables(self) -> List[str]:
        """List of variable names inferred from the ROOT file."""
        if self._element_spec is None:
            return None
        return list(self._dataset.element_spec.keys())

    @property
    def dataset(self) -> tf.data.Dataset:
        """Tensorflow `tf.data.Dataset` object created from the ROOT file."""
        return self._dataset
    
    @property
    def element_spec(self) -> Dict[str, tf.TensorSpec]:
        """`tf.data.Dataset.element_spec` of the `ROOTDataset` object."""
        if self._dataset is None:
            return None
        return self._dataset.element_spec

    @property
    def metadata(self) -> Optional[tf.Tensor]:
        """Tensorflow `tf.Tensor` object containing the metadata of the ROOT file. The metadata is a histogram containing
        the cross section, sum of weights, etc. of the ROOT file. The metadata is read from the ROOT file using the 
        `h_metadata` histogram. If the histogram is not present, the metadata is `None`."""
        return self._metadata
    
    @staticmethod
    def from_root_file(filename: str,
                       tree_name: str = 'NOMINAL',
                       metadata_hist: Optional[str] = 'h_metadata',
                       backend: Literal['pd', 'ak'] = 'pd') -> ROOTDataset:
        """Reads a ROOT file and returns a `ROOTDataset` object. 

        Args:
            filename (str): Path to the ROOT file.
            tree_name (str, optional): Name of the TTree in the ROOT file. Defaults to 'NOMINAL'.
            metadata_hist (str, optional): Name of the histogram containing the metadata. Defaults to 'h_metadata'. Could be `None`.
            backend (str, optional): 'pd' or 'ak'. Backend to use for reading the TTree, 'pd' is faster but consumes more memory. Defaults to 'pd'.

        Returns:
            ROOTDataset: `ROOTDataset` object.
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

            logging.info(f'Done loading file:{filename}')
            dataset = tf.data.Dataset.from_tensor_slices(sample)
        return ROOTDataset(dataset, metadata=metadata if metadata_hist is not None else None)

    @classmethod
    def concat(cls, datasets: List[ROOTDataset]) -> ROOTDataset:
        """Concatenates a list of `ROOTDataset` objects. Data samples are sequentially concatenated using `tf.data.Dataset.concatenate`.

        Args:
            datasets (list[ROOTDataset]): List of `ROOTDataset` objects.

        Raises:
            ValueError: If the variables of the datasets do not match.

        Returns:
            ROOTDataset: Combined `ROOTDataset` object.
        """
        for dataset in datasets:
            if dataset.element_spec != datasets[0].element_spec:
                raise ValueError("Element Spec of datasets do not match")
            if dataset.metadata is not None and dataset.metadata.keys() != datasets[0].metadata.keys():
                raise ValueError("Metadata of datasets do not match")
                
        final_dataset = datasets[0]._dataset
        for ds in datasets[1:]:
            final_dataset = final_dataset.concatenate(ds._dataset)
        if datasets[0].metadata is not None:
            metadata = {key : tf.reduce_sum(tf.stack([ds.metadata[key] for ds in datasets], axis=0), axis=0) for key in datasets[0].metadata}
        else:
            metadata = None
            
        return cls(final_dataset, metadata=metadata)

    @classmethod
    def from_root_files(cls, filenames: Union[List[str], str]) -> ROOTDataset:
        """Reads a list of ROOT files and returns a `ROOTDataset` object. Can also be used to read a single file.

        Args:
            filenames (list[str] or str): List of paths to the ROOT files or a single path to a ROOT file.

        Returns:
            ROOTDataset: `ROOTDataset` object.
        """
        if isinstance(filenames, str):
            filenames = [filenames]
        return cls.concat([cls.from_root_file(filename) for filename in filenames])

    @classmethod
    def load(cls, file: str, element_spec_path: Optional[str] = None,
             metadata_path: Optional[str] = None,) -> ROOTDataset:
        """Loads a `ROOTDataset` object from a saved directory. The saved object is a `tf.data.Dataset` object
        saved using `tf.data.Dataset.save`. The `element_spec` is loaded separately as a pickle object and is used 
        to create the `tf.data.Dataset` object. Defaults to `element_spec` file inside the saved directory.
        Optionally, the `element_spec_path` can be passed as an argument as full path.

        Example:
        Example of creating a `ROOTDataset` object from saved `tf.data.Dataset` object.
        ```python
        import pickle
        import tensorflow as tf

        dataset = tf.data.Dataset.from_tensor_slices({'a': [1, 2, 3], 'b': [4, 5, 6]})
        dataset.save(file_path)
        with open(os.path.join(file_path, 'element_spec'), 'wb') as f:
            pickle.dump(dataset.element_spec, f)

        root_dataset = ROOTDataset.load(file_path)

        ```

        Args:
            file (str): Path to the saved directory.
            element_spec_path (str, optional): Path to the saved `element_spec` as a pickle file. Defaults to `element_spec` file inside the saved directory.


        Returns:
            ROOTDataset: `ROOTDataset` object.
        """

        element_spec_path = os.path.join(
            file, 'element_spec.pkl') if element_spec_path is None else element_spec_path
        with open(element_spec_path, 'rb') as f:
            element_spec = pickle.load(f)
        meta_path = os.path.join(file, 'metadata.pkl') if metadata_path is None else metadata_path
        try:
            with open(meta_path, 'rb') as f:
                metadata = pickle.load(f)
        except FileNotFoundError:
            logging.warning(f"Metadata file not found at {meta_path}")
            logging.warning("Setting metadata to None")
            metadata = None
            
        dataset = tf.data.Dataset.load(
            file, compression='GZIP', element_spec=element_spec)
        return cls(dataset, metadata=metadata)

    def save(self, save_path: str, element_spec_path: Optional[str] = None, 
             metadata_path: Optional[str] = None,
             shard_func: Optional[Callable[[ROOTVariables], tf.Tensor]] = None) -> None:
        """Saves a `ROOTDataset` object to a directory. The saved object is a `tf.data.Dataset` object 
        and the `element_spec` is saved separately as a pickle object saved inside the saved directory.

        Args:
            save_path (str): Path to the directory where the object is to be saved.
            element_spec_path (str, optional): Path to the saved `element_spec` as a pickle file. Defaults to `element_spec` file inside the saved directory.
            shard_func (Callable, optional): Function to shard the dataset. Used as a `shard_func` argument in `tf.data.Dataset.save`. Defaults to `None`.

        Returns:
            None

        """
        element_spec_path = os.path.join(
            save_path, 'element_spec.pkl') if element_spec_path is None else element_spec_path
        metadata_path = os.path.join(
            save_path, 'metadata.pkl') if metadata_path is None else metadata_path
        element_spec = self._dataset.element_spec
        self._dataset.save(save_path, compression='GZIP', shard_func=shard_func)
        with open(element_spec_path, 'wb') as f:
            pickle.dump(element_spec, f)
        if self.metadata is not None:
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)

    def map(self, func: Callable[[ROOTVariables], ROOTVariables]) -> ROOTDataset:
        """Maps a function to the dataset. The function should take a `ROOTVariables` object as input and return a `ROOTVariables` object as output.

        Args:
            func (Callable): Function to be mapped.
            num_parallel_calls (int, optional): Number of parallel calls to use. Defaults to `None`.

        Returns:
            ROOTDataset: Mapped `ROOTDataset` object.
        """
        new_ds = self.dataset.map(func)
        new_ds = new_ds.prefetch(tf.data.AUTOTUNE)
        return ROOTDataset(new_ds, metadata=self.metadata)
    
    def filter(self, func: Callable[[ROOTVariables], tf.Tensor]) -> ROOTDataset:
        """Filters the dataset using a function. The function should take a `ROOTVariables` object as input and return a `tf.Tensor` object as output.

        Args:
            func (Callable): Function to be mapped.
            num_parallel_calls (int, optional): Number of parallel calls to use. Defaults to `None`.

        Returns:
            ROOTDataset: Filtered `ROOTDataset` object.
        """
        new_ds = self.dataset.filter(func)
        return ROOTDataset(new_ds, metadata=self.metadata)
