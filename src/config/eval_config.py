from dataclasses import dataclass
from typing import List, Union

from .config_subclasses import Data


@dataclass
class SubEvalConfig:
    test_sample_cuts: List[str]
    test_names: List[str]


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
    sub_eval: SubEvalConfig
