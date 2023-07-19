"""
Module containing the `JIDENNDataset` dataclass that is a wrapper for a TensorFlow dataset that allows for easy adding and processing of dataset files.
It contains all the necessary tools to perform a preprocessing of the jet dataset for training. 
"""
from __future__ import annotations
import tensorflow as tf
import pandas as pd
from dataclasses import dataclass
from typing import Union, Literal, Callable, Dict, Tuple, List, Optional, Any
import os
import pickle
#
import jidenn.config.config as config
from jidenn.data.string_conversions import Cut, Expression
from jidenn.evaluation.plotter import plot_data_distributions


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


@tf.function
def dict_to_stacked_array(data: Union[ROOTVariables, Tuple[ROOTVariables, ROOTVariables]], label: int, weight: Optional[float] = None) -> Tuple[Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]], int, Union[float, None]]:
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
        interaction = tf.stack([data[1][var] for var in data[1]], axis=-1)
        interaction = tf.where(tf.math.logical_or(tf.math.is_inf(interaction), tf.math.is_nan(interaction)),
                               tf.zeros_like(interaction), interaction)
        if weight is None:
            return (tf.stack([data[0][var] for var in data[0]], axis=-1), interaction), label
        return (tf.stack([data[0][var] for var in data[0]], axis=-1), interaction), label, weight
    else:
        if weight is None:
            return tf.stack([data[var] for var in data.keys()], axis=-1), label
        return tf.stack([data[var] for var in data.keys()], axis=-1), label, weight


@dataclass
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

    Args:
        variables (List[str]): The list of variables to be used in the dataset.
        target (str, optional): The name of the target variable. Defaults to `None`.
        weight (str, optional): The name of the weight variable. Defaults to `None`.


    """
    variables: Optional[List[str]] = None
    """The configuration dataclass of the variables to be used in the dataset. 
        If `None`, the variables are set automatically during loading with JIDENNDataset.load()."""
    target: Optional[str] = None
    """The name of the target variable. `None` if no target variable is used."""
    weight: Optional[str] = None
    """The name of the weight variable. `None` if no weight variable is used."""

    def __post_init__(self):
        self._dataset = None
        self._element_spec = None

    def load_dataset(self, file: str) -> JIDENNDataset:
        """Loads a dataset from a file. The dataset is stored in the `tf.data.Dataset` format.
        The `element_spec` is loaded from the `element_spec` file inside the dataset directory.    
        Alternatively, the `element_spec` can be loaded manually using the `load_element_spec` method.

        Args:
            file (str): The path to the dataset directory.

        Returns:
            JIDENNDataset: The JIDENNDataset object with set dataset and `element_spec`.

        """
        if self.element_spec is None:
            element_spec_file = os.path.join(file, 'element_spec')
            jidenn_dataset = self.load_element_spec(element_spec_file)
        else:
            jidenn_dataset = self
        dataset = tf.data.Dataset.load(
            file, compression='GZIP', element_spec=jidenn_dataset.element_spec)
        return jidenn_dataset._set_dataset(dataset)

    @staticmethod
    def load(path: str, element_spec_path: Optional[str] = None) -> JIDENNDataset:
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

        if element_spec_path is None:
            element_spec_path = os.path.join(path, 'element_spec')
        with open(element_spec_path, 'rb') as f:
            element_spec = pickle.load(f)

        if isinstance(element_spec, dict):
            variables = list(element_spec.keys())

        elif isinstance(element_spec[0], dict):
            variables = list(element_spec[0].keys())

        else:
            raise ValueError('Element spec is not a dictionary.')

        return JIDENNDataset(variables=variables).load_dataset(path)

    def save_dataset(self, file: str, num_shards: Optional[int] = None) -> None:
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
                          shard_func=random_shards if num_shards is not None else None)
        with open(os.path.join(file, 'element_spec'), 'wb') as f:
            pickle.dump(self.dataset.element_spec, f)

    def load_element_spec(self, file: str) -> JIDENNDataset:
        """Loads the `element_spec` from a file. The `element_spec` is a pickled dictionary of `tf.TensorSpec` or `tf.RaggedTensorSpec` objects.

        Args:
            file (str): The path to the `element_spec` file.
        Returns:
            JIDENNDataset: The JIDENNDataset object with the `element_spec` set.
        """
        with open(file, 'rb') as f:
            element_spec = pickle.load(f)
        return self._set_element_spec(element_spec)

    def create_variables(self, cut: Optional[Cut] = None, map_dataset: Optional[Callable[[ROOTVariables], ROOTVariables]] = None) -> JIDENNDataset:
        """Creates a 'tf.data.Dataset' from selected variables and creates labels and weights.
        The variables are selected according to the `variables` loaded from config.
        the `target` and `weight` class variables are used to create labels and weights from the `ROOTVariables`.

        Optionally, a `Cut` can be applied to the dataset. It is done **before** the variables are selected.
        The `map_dataset` function can be used to apply a function to the dataset before the variables are selected.
        It could be used to create new variables from the existing ones.

        Args:
            cut (jidenn.data.utils.Cut.Cut, optional): The `Cut` object to be applied to the dataset. Defaults to `None`. 
            map_dataset (Callable[[ROOTVariables], ROOTVariables], optional): The function to be applied to the dataset using `tf.data.Dataset.map`. Defaults to `None`.

        Raises:
            ValueError: If the dataset is not loaded yet.

        Returns:
            JIDENNDataset: The JIDENNDataset object with the signature of `(ROOTVariables, label, weight)`.
        """
        if self.dataset is None:
            raise ValueError('Dataset not loaded yet.')
        if map_dataset is not None:
            dataset = self.dataset.map(map_dataset)
        else:
            dataset = self.dataset

        dataset = dataset.filter(cut) if cut is not None else dataset
        dataset = dataset.map(self._var_picker)
        return self._set_dataset(dataset)

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

        if self.weight is not None:
            @tf.function
            def remap_label(x, y, w):
                return x, label_mapping(y), w
        else:
            @tf.function
            def remap_label(x, y):
                return x, label_mapping(y)

        dataset = self.dataset.map(remap_label)
        return self._set_dataset(dataset)

    @property
    def dataset(self) -> Union[tf.data.Dataset, None]:
        """The `tf.data.Dataset` object or `None` if the dataset is not set yet."""
        return self._dataset

    @property
    def element_spec(self) -> Union[Dict[str, Union[tf.TensorSpec, tf.RaggedTensorSpec]], None]:
        """The `element_spec` of the dataset or `None` if the dataset is not set yet."""
        return self._element_spec

    def _set_element_spec(self, element_spec: Dict[str, Union[tf.TensorSpec, tf.RaggedTensorSpec]]) -> JIDENNDataset:
        jidenn_dataset = JIDENNDataset(variables=self.variables,
                                       target=self.target,
                                       weight=self.weight)
        self._element_spec = element_spec
        self._dataset = self.dataset
        return jidenn_dataset

    def _set_dataset(self, dataset: Union[tf.data.Dataset, None]) -> JIDENNDataset:
        jidenn_dataset = JIDENNDataset(variables=self.variables,
                                       target=self.target,
                                       weight=self.weight)
        jidenn_dataset._dataset = dataset
        jidenn_dataset._element_spec = dataset.element_spec
        return jidenn_dataset

    def set_dataset(self, dataset: tf.data.Dataset, element_spec: Dict[str, Union[tf.TensorSpec, tf.RaggedTensorSpec]]) -> JIDENNDataset:
        """Sets the `tf.data.Dataset` object and the `element_spec` of the dataset.

        Args:
            dataset (tf.data.Dataset): The `tf.data.Dataset` object consisting of `ROOTVariables`.
            element_spec (Dict[str, Union[tf.TensorSpec, tf.RaggedTensorSpec]]): The `element_spec` of the dataset.

        Returns:
            JIDENNDataset: The JIDENNDataset object with the `dataset` and `element_spec` set.
        """
        jidenn_dataset = JIDENNDataset(variables=self.variables,
                                       target=self.target,
                                       weight=self.weight)
        jidenn_dataset._dataset = dataset
        jidenn_dataset._element_spec = element_spec
        return jidenn_dataset

    @property
    def _var_picker(self):
        @tf.function
        def _pick_variables(sample: ROOTVariables) -> Union[Tuple[ROOTVariables, tf.RaggedTensor, tf.RaggedTensor], ROOTVariables, Tuple[ROOTVariables, tf.RaggedTensor]]:

            new_sample = {var: Expression(var)(sample)
                          for var in self.variables}

            if self.target is None:
                return new_sample
            if self.weight is None:
                return new_sample, Expression(self.target)(sample)
            else:
                return new_sample, Expression(self.target)(sample), Expression(self.weight)(sample)
        return _pick_variables

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
        return self._set_dataset(dataset)

    @staticmethod
    def combine(datasets: List[JIDENNDataset], weights: List[float]) -> JIDENNDataset:
        """Combines multiple datasets into one dataset. The samples are interleaved and the weights are used to sample from the datasets.

        Args:
            datasets (List[JIDENNDataset]): List of datasets to combined. All `JIDENNDataset.dataset`s must be set and have the same `element_spec`.
            weights (List[float]): List of weights for each dataset. The weights are used to sample from the datasets.

        Returns:
            JIDENNDataset: Combined `JIDENNDataset` object.
        """
        dataset = tf.data.Dataset.sample_from_datasets(
            [dataset.dataset for dataset in datasets], weights=weights)
        jidenn_dataset = JIDENNDataset(
            datasets[0].variables, datasets[0].target, datasets[0].weight)
        return jidenn_dataset._set_dataset(dataset)

    def apply(self, func: Callable[[tf.data.Dataset], tf.data.Dataset]) -> JIDENNDataset:
        """Applies a function to the dataset.

        Args:
            func (Callable[[tf.data.Dataset], tf.data.Dataset]): Function to apply to the dataset.

        Returns:
            JIDENNDataset: The JIDENNDataset object with the dataset modified by the function.

        """
        if self.dataset is None:
            raise ValueError('Dataset not loaded yet.')
        dataset = func(self.dataset)
        return self._set_dataset(dataset)

    def create_train_input(self, func: Callable[[ROOTVariables], Union[ROOTVariables, Tuple[ROOTVariables, ROOTVariables]]]) -> JIDENNDataset:
        """Creates a training input from the dataset using the `func` function. The function must take a `ROOTVariables` object and return a `ROOTVariables` object.
        The output of the function is of the form Dict[str, tf.Tensor] or Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]] (optionally aslo tf.RaggedTensor).


        Args:
            func (Callable[[ROOTVariables], Union[ROOTVariables, Tuple[ROOTVariables, ROOTVariables]]]): Function to apply to the data to create the training input.

        Raises:
            ValueError: If the dataset is not loaded yet.

        Returns:
            JIDENNDataset: The JIDENNDataset object  with signature `((ROOTVariables, ROOTVariables), ...)` or `(ROOTVariables, ...)`.
        """
        if self.dataset is None:
            raise ValueError('Dataset not loaded yet.')

        @tf.function
        def input_wrapper(data, label, w=None):
            return func(data), label
        dataset = self.dataset.map(input_wrapper)
        return self._set_dataset(dataset)

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

        try:
            import tensorflow_datasets as tfds
        except ImportError:
            raise ImportError(
                'Please install tensorflow_datasets to use this function. Use `pip install tensorflow_datasets`.')
        if self.dataset is None:
            raise ValueError('Dataset not loaded yet.')

        if isinstance(self.element_spec, tuple) and variables is None:
            @tf.function
            def tuple_to_dict(data, label, weight=None):
                if isinstance(data, tuple):
                    data = {**data[0], **data[1]}
                data = {**data, 'label': label, 'weight': weight}
                return data

        elif isinstance(self.element_spec, tuple) and variables is not None:
            @tf.function
            def tuple_to_dict(data, label, weight=None):
                if isinstance(data, tuple):
                    data = {**data[0], **data[1]}
                data = {**data, 'label': label, 'weight': weight}
                return {k: data[k] for k in variables + ['label', 'weight']}

        elif isinstance(self.element_spec, dict) and variables is not None:
            @tf.function
            def tuple_to_dict(data):
                return {k: data[k] for k in variables}

        elif isinstance(self.element_spec, dict) and variables is None:
            @tf.function
            def tuple_to_dict(data):
                return data

        else:
            raise ValueError('The dataset must be a tuple or a dict.')

        dataset = self.dataset.map(tuple_to_dict)
        df = tfds.as_dataframe(dataset)
        df = df.rename(lambda x: x.replace('/', '.'), axis='columns')
        return df

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
        return self._set_dataset(dataset)

    def get_prepared_dataset(self,
                             batch_size: int,
                             assert_length: bool = False,
                             shuffle_buffer_size: Optional[int] = None,
                             take: Optional[int] = None,
                             map_func: Optional[Callable[[Union[ROOTVariables, Tuple[ROOTVariables, ROOTVariables]], Any], Tuple[Union[ROOTVariables, Tuple[ROOTVariables, ROOTVariables]], Any]]] = None) -> tf.data.Dataset:
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
        if map_func is not None:
            dataset = self.dataset.map(map_func)
        else:
            dataset = self.dataset.map(dict_to_stacked_array)
        dataset = dataset.shuffle(
            shuffle_buffer_size) if shuffle_buffer_size is not None else dataset
        if take is not None:
            dataset = dataset.take(take)
            dataset = dataset.apply(tf.data.experimental.assert_cardinality(
                take)) if assert_length else dataset
        dataset = dataset.apply(
            tf.data.experimental.dense_to_ragged_batch(batch_size))
        # dataset = dataset.ragged_batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def plot_data_distributions(self,
                                folder: str,
                                variables: Optional[List[str]] = None,
                                hue_variable: Optional[str] = None,
                                named_labels: Optional[Dict[int, str]] = None,
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
        df = self.to_pandas(variables)
        plot_data_distributions(df, folder=folder, named_labels=named_labels,
                                xlabel_mapper=xlabel_mapper, hue_variable=hue_variable)
