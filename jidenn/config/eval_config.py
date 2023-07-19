"""
Configuration for evaluation of a model.
The `jidenn.config.config.Data` configuration is equivalent to the one used for training.
Evaulation of a model can be done for several subsamples of the test data.
They are specified by different cuts on the test data, such as the `jets_pt` or `jets_eta`.
These binning based on cuts are specified in the `jidenn.config.eval_config.Binning` configuration.
"""
from dataclasses import dataclass
from typing import List, Optional, Literal, Tuple, Union

from .config import Data


@dataclass
class Binning:
    """Configuration for binning a continuous variable.

    Args:
        variable (str): Name of the variable to bin.
        bins (int): Number of bins to use.
        max_bin (Union[float, int]): Maximum value of the variable. Values above this will be put in the last bin.
        min_bin (Union[float, int]): Minimum value of the variable. Values below this will be put in the first bin.
        log_bin_base (Optional[int]): If not `None`, use logarithmic binning with the specified base.
    """
    variable: str
    bins: int
    max_bin: Union[float, int]
    min_bin: Union[float, int]
    log_bin_base: Optional[int]


@dataclass
class EvalConfig:
    """Configuration for evaluating a machine learning model.

    Args:
        data (jidenn.config.config.Data): Configuration for the data.
        logdir (str): Path to the log directory of the evaluation. 
        models_path (str): Path to the directory where the models to evaluate are saved. 
            Inside this directory, there should be a subdirectory for each model with 
            the same name as in the `model_names` list. Each of these subdirectories must
            include a subdirectory `model` containing the saved model.
        model_names (List[str]): List of names of the models to evaluate.
        model_input_types (List[str]): List of types of input data expected by the models.
        seed (int): Seed for reproducibility.
        draw_distribution (int, optional): If not `None`, draw the distribution of the train input
            variables for the first `draw_distribution` events.
        test_subfolder (str): Subfolder of the data folder to use for evaluation. 
            One of 'test', 'train', 'dev'.
        batch_size (int): Batch size for evaluation.
        take (int): Number of data samples to use for evaluation. 
        binning (jidenn.config.eval_config.Binning): Configuration for binning of test data.
        working_point_path (Optional[str]): Path to a folder where for each model there is a subfolder with the name of the model.
            contains a csv file with the name `threshold_file_name` containing a column with the name `threshold_var_name`.
            Folder structure: `threshold_path`/`model_name`/`threshold_file_name`.
        working_point_file_name (Optional[str]): Name of the threshold file.
        metrics_to_plot (List[str]): List of metrics to plot. See `jidenn.evaluation.evaluation_metrics.get_metrics` for options.
        ylims (List[List[float]]): List of y-axis limits for each metric plot.
        reference_model (str): Name of the model to use as reference for ratio plots.
        threads (Optional[int]): Number of threads to use for evaluation. If `None`, single-threaded evaluation is used.
        validation_plots_in_bins (bool): If `True`, create validation plots for each bin in the binning.
    """
    data: Data
    logdir: str
    models_path: str
    model_names: List[str]
    model_input_types: List[str]
    save_path: str
    seed: int
    draw_distribution: Optional[int]
    test_subfolder: str
    batch_size: int
    take: int
    binning: Binning
    working_point_path: Optional[str]
    working_point_file_name: Optional[str]
    metrics_to_plot: List[str]
    reference_model: str
    ylims: List[List[float]]
    threads: Optional[int]
    validation_plots_in_bins: bool
