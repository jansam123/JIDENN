from dataclasses import dataclass
from typing import List, Union, Literal

from .config_subclasses import Data


@dataclass
class Binning:
    test_sample_cuts: List[str]
    test_names: List[str]
    type: Literal["perJet", "perJetTuple", "perEvent"]


@dataclass
class EvalConfig:
    logdir: str
    base_logdir: str
    data: Data
    seed: int
    draw_distribution: Union[int, None]
    test_subfolder: str
    model_dir: str
    batch_size: int
    take: int
    feature_importance: bool
    model: str
    binning: Binning
    threshold: float
    old: bool
    include_base: bool
    input_type = Literal['highlevel',
                         'highlevel_constituents',
                         'constituents',
                         'relative_constituents',
                         'interaction_constituents',
                         'deepset_constituents']
