from typing import Optional, List, Dict, Tuple, Union, Callable, Literal
import tensorflow as tf
import numpy as np
import pandas as pd
import logging
import time
import hashlib
import pickle
import os
#
from jidenn.data.JIDENNDataset import JIDENNDataset, ROOTVariables
from jidenn.data.TrainInput import input_classes_lookup
from .evaluation_metrics import calculate_metrics
from .WorkingPoint import BinnedVariable
from multiprocessing import Pool


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
    if dataset.weight is None:
        @tf.function
        def add_to_dict(data_label: Tuple[ROOTVariables, tf.Tensor], score: tf.Tensor) -> Tuple[ROOTVariables, tf.Tensor]:
            data, label = data_label[0].copy(), data_label[1]
            data[score_name] = score
            return data, label
    else:
        @tf.function
        def add_to_dict(data_label: Tuple[ROOTVariables, tf.Tensor], score: tf.Tensor) -> Tuple[ROOTVariables, tf.Tensor, tf.Tensor]:
            data, label, weight = data_label[0].copy(), data_label[1], data_label[2]
            data[score_name] = score
            return data, label, weight

    score_dataset = tf.data.Dataset.from_tensor_slices(score)
    new_dataset = tf.data.Dataset.zip((dataset.dataset, score_dataset))
    new_dataset = new_dataset.map(add_to_dict)

    return JIDENNDataset(dataset=new_dataset, element_spec=dataset.element_spec, metadata=dataset.metadata, target=dataset.target, weight=dataset.weight)


def _calculate_metrics_in_bin(x):
    y, score_variable, threshold, validation_plotter, weights_variable = x
    inter, x = y
    if x.empty:
        return
    if len(x['label'].unique()) < 2:
        return

    if validation_plotter is not None:
        validation_plotter(x)
    ret = calculate_metrics(x['label'], x[score_variable], threshold=threshold,
                            weights=x[weights_variable] if weights_variable is not None else None)
    ret['num_events'] = len(x)
    ret['eff_num_events'] = np.sum(x[weights_variable])**2/np.sum(x[weights_variable]**2) if weights_variable is not None else len(x)
    ret['eff_num_events_q'] = np.sum(x.query('label==1')[weights_variable])**2/np.sum(x.query('label==1')[weights_variable]**2) if weights_variable is not None else len(x.query('label==1'))
    ret['eff_num_events_g'] = np.sum(x.query('label==0')[weights_variable])**2/np.sum(x.query('label==0')[weights_variable]**2) if weights_variable is not None else len(x.query('label==0'))
    ret['bin'] = inter
    return ret


def calculate_binned_metrics(df: pd.DataFrame,
                             binned_variable: str,
                             score_variable: str,
                             bins: Union[List[Union[float, int]], np.ndarray],
                             weights_variable: Optional[str] = None,
                             validation_plotter: Optional[Callable[[pd.DataFrame], None]] = None,
                             threshold: Union[BinnedVariable, float] = 0.5,
                             threads: Optional[int] = None) -> pd.DataFrame:
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
            pd.Interval (e.g. 'm(0.5, 1.0]') and a column with the name specified in `threshold_name` containing
            the threshold values for each bin. Default is 0.5.
        threshold_name (str, optional): Name of the column containing the threshold values in the threshold
            DataFrame. Only used if a DataFrame is provided as the threshold argument. Default is None.

    Returns:
        pd.DataFrame: DataFrame containing the calculated metrics for each bin.

    """

    df['bin'] = pd.cut(df[binned_variable], bins=bins) if not isinstance(
        bins, BinnedVariable) else pd.cut(df[binned_variable], bins=threshold.bins)
    grouped_metrics = df.groupby('bin')
    threshold_values = threshold.values if isinstance(threshold, BinnedVariable) else [threshold] * len(grouped_metrics)
    if weights_variable is not None:
        args = [(x, score_variable, th, validation_plotter, weights_variable)
                for x, th in zip(grouped_metrics, threshold_values)]
    else:
        args = [(x, score_variable, th, validation_plotter, None) for x, th in zip(grouped_metrics, threshold_values)]

    if threads is not None and threads > 1:
        with Pool(threads) as pool:
            metrics = pool.map(_calculate_metrics_in_bin, args)
    else:
        metrics = map(_calculate_metrics_in_bin, args)

    metrics = [x for x in metrics if x is not None]
    metrics = pd.DataFrame(metrics)
    return metrics


def benchmark(func):
    num_gpus = len(tf.config.list_physical_devices("GPU"))

    def wrapper(*args, **kwargs):
        [tf.config.experimental.reset_memory_stats(f'GPU:{i}') for i in range(num_gpus)]
        start = time.time()
        ret = func(*args, **kwargs)
        stop = time.time()
        max_memory = sum([tf.config.experimental.get_memory_info(f'GPU:{i}')['peak'] for i in range(num_gpus)])
        total_time = stop - start
        return ret, total_time, max_memory
    return wrapper


@benchmark
def predict(model: tf.keras.Model, dataset: tf.data.Dataset):
    score = model.predict(dataset).ravel()
    return score


def evaluate_multiple_models(model_paths: List[str],
                             model_names: List[str],
                             dataset: JIDENNDataset,
                             model_input_name: List[Literal['highlevel',
                                                            'highlevel_constituents',
                                                            'constituents',
                                                            'relative_constituents',
                                                            'interaction_constituents']],
                             batch_size: int,
                             take: Optional[int] = None,
                             score_name: str = 'score',
                             log: Optional[logging.Logger] = None,
                             custom_objects: Optional[Dict[str, Callable]] = None,
                             checkpoint_path: Optional[str] = None,
                             distribution_drawer: Optional[Callable[[JIDENNDataset, str], None]] = None) -> Tuple[JIDENNDataset, pd.DataFrame]:
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
    tech_info = []
    for input_type in set(model_input_name):
        train_input_class = input_classes_lookup(input_type)
        train_input_class = train_input_class()
        model_input = tf.function(func=train_input_class)
        ds = dataset.remap_data(model_input)
        if distribution_drawer is not None and 'interaction_constituents' not in input_type:
            log.info(f'----- Drawing data distribution for: {input_type}') if log is not None else None
            input_type_hash = hashlib.sha256(input_type.encode('utf-8')).hexdigest()
            if checkpoint_path is not None:
                try:
                    with open(os.path.join(checkpoint_path, f'{input_type_hash}'), 'rb') as f:
                        pickle.load(f)
                    log.info(f'----- Data distribution already drawn for: {input_type}') if log is not None else None
                except FileNotFoundError:
                    distribution_drawer(ds, input_type)
                    with open(os.path.join(checkpoint_path, f'{input_type_hash}'), 'wb') as f:
                        pickle.dump(True, f)
            else:
                distribution_drawer(ds, input_type)
            
        ds = ds.get_prepared_dataset(batch_size=batch_size,
                                     ragged=False if 'gnn' in input_type else True,
                                     take=take)

        # iterate over all models with the same input type
        idxs = np.array(model_input_name) == input_type
        for model_path, model_name in zip(np.array(model_paths)[idxs], np.array(model_names)[idxs]):
            model_name_hash = hashlib.sha256(model_name.encode('utf-8')).hexdigest()
            if checkpoint_path is not None:
                try:
                    with open(os.path.join(checkpoint_path, f'{model_name_hash}.pkl'), 'rb') as f:
                        score = pickle.load(f)
                    with open(os.path.join(checkpoint_path, f'{model_name_hash}_tech.pkl'), 'rb') as f:
                        tech_info_current = pickle.load(f)
                    log.info(f'----- Score already calculated for: {model_name}') if log is not None else None
                    dataset = add_score_to_dataset(dataset, score, f'{model_name}_{score_name}')
                    tech_info.append(tech_info_current)
                    continue
                except FileNotFoundError:
                    pass
            log.info(f'----- Loading model: {model_name}') if log is not None else None
            start = time.time()
            model : tf.keras.Model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            model.compile()
            stop = time.time()
            log.info(f'----- Loading model took: {stop-start:.2f} s') if log is not None else None
            log.info(f'----- Predicting with model: {model_name}') if log is not None else None
            start = time.time()
            score, total_time, max_memory = predict(model, ds)
            stop = time.time()
            log.info(f'----- Predicting took: {total_time:.2f} s') if log is not None else None
            log.info(f'----- Max memory used: {max_memory*1e-6:.2f} MB') if log is not None else None
            tech_info_current = pd.DataFrame({'time': total_time, 'memory': max_memory*1e-6}, index=[model_name])
            if checkpoint_path is not None:
                with open(os.path.join(checkpoint_path, f'{model_name_hash}.pkl'), 'wb') as f:
                    pickle.dump(score, f)
                with open(os.path.join(checkpoint_path, f'{model_name_hash}_tech.pkl'), 'wb') as f:
                    pickle.dump(tech_info_current, f)
            dataset = add_score_to_dataset(dataset, score, f'{model_name}_{score_name}')
            tech_info.append(tech_info_current) 

    return dataset, pd.concat(tech_info)
