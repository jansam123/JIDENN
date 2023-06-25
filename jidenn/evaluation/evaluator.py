from typing import Optional, List, Dict, Tuple, Union, Callable
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
import logging
import time
#
from jidenn.data.JIDENNDataset import JIDENNDataset, ROOTVariables
from jidenn.data.TrainInput import input_classes_lookup
from .evaluation_metrics import calculate_metrics


def add_score_to_dataset(dataset: JIDENNDataset,
                         score: np.ndarray,
                         score_name: str = 'score') -> JIDENNDataset:
    """Add a score array to a JIDENNDataset. Score could me any variable that is not part of the original dataset.
    It is important that the score array has the same length as the dataset. This is useful for adding the output of a
    ML model to the original dataset before the trining input is created.

    I/O Example:
    ```python
    example_input_element = {'E': 1.0, 'eta': 1.0, 'pt': 1.0, 'phi': 1.0, 'label': 1, 'num': 1, 'event': 1, 'mu': 1.0, 'corr_mu': 1.0}
    example_output_element = {'E': 1.0, 'eta': 1.0, 'pt': 1.0, 'phi': 1.0, 'label': 1, 'num': 1, 'event': 1, 'mu': 1.0, 'corr_mu': 1.0, 'score': 0.5}
    ```

    Args:
        dataset (JIDENNDataset): JIDENNDataset to add the score to.
        score (np.ndarray): Array containing the score values to add.
        score_name (str, optional): Name of the score variable inside the new dataset. Default is 'score'.

    Returns:
        JIDENNDataset: JIDENNDataset with the score added. Its elements will have the same structure as the original,
        i.e. a dictionary with the same kay-value pairs with one additional key-value pair for the score `{score_name: score[i]}`.

    """
    @tf.function
    def add_to_dict(data_label: Tuple[ROOTVariables, tf.Tensor], score: tf.Tensor) -> Tuple[ROOTVariables, tf.Tensor]:
        data, label = data_label[0].copy(), data_label[1]
        data[score_name] = score
        return data, label

    score_dataset = tf.data.Dataset.from_tensor_slices(score)
    dataset = tf.data.Dataset.zip((dataset.dataset, score_dataset))
    dataset = dataset.map(add_to_dict)
    variables = list(dataset.element_spec[0].keys())

    return JIDENNDataset(variables).set_dataset(dataset, element_spec=dataset.element_spec)


def calculate_binned_metrics(df: pd.DataFrame,
                             binned_variable: str,
                             score_variable: str,
                             bins: Union[List[Union[float, int]], np.ndarray],
                             validation_plotter: Optional[Callable[[pd.DataFrame], None]] = None,
                             threshold: Union[pd.DataFrame, float] = 0.5,
                             threshold_name: Optional[str] = None) -> pd.DataFrame:
    """Calculate metrics for a binary classification problem binned by a continuous variable.

    Example pd.DataFrame structure:
    ```python
    df = pd.DataFrame({'label': [0, 1, 0, 1, 0, 1, 0, 1],
                        'score': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, .7, .8],
                        'jets_pt': [1, 2, 3, 4, 5, 6, 7, 8]})
    score_variable = 'score'
    binning_variable = 'jets_pt'
    bins = [2, 5, 7]
    binned_metrics = calculate_binned_metrics(df=df,
                                                binned_variable=binned_variable,
                                                score_variable=score_variable,
                                                bins=bins)
    print(binned_metrics)
    # Output:
    #    accuracy  signal_efficiency background_efficiency num_events           bin
    # 1  0.500000           0.500000              0.500000          2   '(2.0, 5.0]'
    # 2  0.666667           0.666667              0.666667          3   '(5.0, 7.0]'

    ```

    Args:
        df (pd.DataFrame): DataFrame containing columns `label`, `score_variable` and `binned_variable`.
        binned_variable (str): Name of the column containing the continuous variable to bin.
        score_variable (str): Name of the column containing the model scores.
        bins (Union[List[Union[float, int]], np.ndarray]): List or array of bin edges to use.
        validation_plotter (Callable[[pd.DataFrame], None], optional): Function to plot validation data
            for each bin (confusion matrix, ROC, score outputs histogram,...). Default is None.
        threshold (Union[pd.DataFrame, float], optional): Threshold value for the binary classification.
            If a DataFrame is provided, it should contain a 'bin' column with string representation of 
            pd.Interval (e.g. '(0.5, 1.0]') and a column with the name specified in `threshold_name` containing
            the threshold values for each bin. Default is 0.5.
        threshold_name (str, optional): Name of the column containing the threshold values in the threshold
            DataFrame. Only used if a DataFrame is provided as the threshold argument. Default is None.

    Returns:
        pd.DataFrame: DataFrame containing the calculated metrics for each bin.

    """

    df['bin'] = pd.cut(df[binned_variable], bins=bins)

    def calculator(x):
        if isinstance(threshold, pd.DataFrame) and threshold_name is not None:
            threshold_val = threshold.loc[threshold['bin'] == str(x['bin'].iloc[0]), threshold_name]
            threshold_val = float(threshold_val.iloc[0])
        else:
            threshold_val = threshold
        if validation_plotter is not None:
            validation_plotter(x)
        ret = calculate_metrics(x['label'], x[score_variable], threshold=threshold_val)
        ret['num_events'] = len(x)
        return ret

    metrics = df.groupby('bin').apply(calculator)
    bin_intervals = metrics.index
    metrics = metrics.reset_index(drop=True).apply(pd.Series)
    metrics['bin'] = bin_intervals
    metrics = metrics.dropna()

    return metrics


def evaluate_multiple_models(model_paths: List[str],
                             model_names: List[str],
                             dataset: JIDENNDataset,
                             model_input_name: List[str],
                             batch_size: int,
                             take: Optional[int] = None,
                             score_name: str = 'score',
                             log: Optional[logging.Logger] = None,
                             custom_objects: Optional[Dict[str, Callable]] = None,
                             distribution_drawer: Optional[Callable[[JIDENNDataset], None]] = None) -> JIDENNDataset:
    """Evaluate multiple Keras models on a JIDENNDataset. The explicit training inputs are created automatically
    from the JIDENNDataset. Input type for each model is deduced from the `model_input_name` argument. The order of 
    evaluation is **NOT** determined by the `model_names` argument. The iteration order is given by the unigue values
    in `model_input_name`, to reduce the number of times the dataset is prepared.

    Args:
        model_paths (List[str]): List of paths to the Keras model files. They will be loaded with
            `tf.keras.models.load_model(model_path, custom_objects=custom_objects)`.
        model_names (List[str]): List of names for each model.
        dataset (JIDENNDataset): JIDENNDataset to evaluate the models on and to add the scores to.
        model_input_name (List[str]): List of input names for each model. See `jidenn.data.TrainInput.input_classes_lookup`
            for options.
        batch_size (int): Batch size to use for the evaluation.
        take (int, optional): Number of events to evaluate. If not provided, all events will be used. Default is None.
        score_name (str, optional): Name of the score variable to add to the dataset. Default is 'score'. For each model,
            the score will be added with the name `f'{model_name}_{score_name}'`.
        log (logging.Logger, optional): Logger to use for logging messages and evaluation/loading times. Default is None.
        custom_objects (Dict[str, Callable], optional): Dictionary of custom objects to use when loading the models.
            Passed to `tf.keras.models.load_model(model_path, custom_objects=custom_objects)`. Default is None.
        distribution_drawer (Callable[[JIDENNDataset], None], optional): Function to plot the data distribution of 
            the input variables which are automatically created with the `jidenn.data.TrainInput` class. Default is None.

    Returns:
        JIDENNDataset: JIDENNDataset with the scores added.

    """

    # iterate over all input types to reduce the number of times the dataset is prepared
    log.info(f'Batches will be of size: {batch_size}, total number of events: {take}') if log is not None else None
    for input_type in set(model_input_name):
        train_input_class = input_classes_lookup('constituents')
        train_input_class = train_input_class()
        model_input = tf.function(func=train_input_class)
        ds = dataset.create_train_input(model_input)
        if distribution_drawer is not None:
            log.info(f'----- Drawing data distribution for: {input_type}') if log is not None else None
            distribution_drawer(ds)
        ds = ds.get_prepared_dataset(batch_size=batch_size, take=take)

        # iterate over all models with the same input type
        idxs = np.array(model_input_name) == input_type
        for model_path, model_name in zip(np.array(model_paths)[idxs], np.array(model_names)[idxs]):
            log.info(f'----- Loading model: {model_name}') if log is not None else None
            start = time.time()
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            stop = time.time()
            log.info(f'----- Loading model took: {stop-start:.2f} s') if log is not None else None
            log.info(f'----- Predicting with model: {model_name}') if log is not None else None
            start = time.time()
            score = model.predict(ds).ravel()
            stop = time.time()
            log.info(f'----- Predicting took: {stop-start:.2f} s') if log is not None else None
            dataset = add_score_to_dataset(dataset, score, f'{model_name}_{score_name}')

    return dataset
