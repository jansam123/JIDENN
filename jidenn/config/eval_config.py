"""
Configuration for evaluation of a model.
The `jidenn.config.config.Data` configuration is equivalent to the one used for training.
Evaulation of a model can be done for several subsamples of the test data.
They are specified by different cuts on the test data, such as the `jets_pt` or `jets_eta`.
These binning based on cuts are specified in the `jidenn.config.eval_config.Binning` configuration.
"""
from dataclasses import dataclass
from typing import List, Optional, Literal

from .config import Data


@dataclass
class Binning:
    """Configuration for binning of test data.
    Each bin is defined by a cut on the test data.

    Args:
        cuts (List[str]): Cuts on the test data performing the binning. 
            See `jidenn.data.string_conversions.Cut` for more details.
        cut_names (List[str]): Names of the cuts displayed in the plots and subfolders.
    """
    cuts: List[str]
    cut_names: List[str]


@dataclass
class EvalConfig:
    """Configuration for evaluation of a model.
    
    Args:
        base_logdir (str): Base path to the log directory of the training. 
        eval_logs (str): Subfolder of the base log directory for the evaluation.
        logdir (str): Path to the log directory of the evaluation. Could be set manually,
            but using `${base_logdir}/${eval_logs}` is recommended, 
            as it creates a unique folder for each training session inside the `base_logdir`.
        data (jidenn.config.config.Data): Configuration for the test data.
        seed (int): Seed for reproducibility.
        draw_distribution (int, optional): Draw the distribution of the train input variables 
            for the first `draw_distribution` events. If `None`, no distribution is drawn.
        test_subfolder (str): Subfolder of the data folder to use, One of 'test', 'train', 'dev'.
        model_dir (str): Path to the saved model directory, using `tf.keras.Model.save`.
        batch_size (int): Batch size for evaluation.
        take (int): Number of data samples to use for evaluation. 
        feature_importance (bool): If `True`, compute the feature importance of the model.
        binning (jidenn.config.eval_config.Binning): Configuration for binning of test data.
        threshold (float): Threshold for distinguishing quarks and gluons.
        include_base (bool): If `False`, and the binning is applied, the evaluation without
            binning is not performed.
        input_type (str): Input type of the model. One of: 'highlevel', 'highlevel_constituents',
            'constituents', 'relative_constituents', 'interaction_constituents'.
        
            
    
    """
    logdir: str
    base_logdir: str
    data: Data
    seed: int
    draw_distribution: Optional[int]
    test_subfolder: str
    model_dir: str
    batch_size: int
    take: int
    feature_importance: bool
    binning: Optional[Binning]
    threshold: float
    include_base: bool
    input_type = Literal['highlevel',
                         'highlevel_constituents',
                         'constituents',
                         'relative_constituents',
                         'interaction_constituents']
