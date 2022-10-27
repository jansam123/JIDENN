from dataclasses import dataclass
from .config_subclasses import Data, Dataset

@dataclass
class EvalConfig:
    logdir: str
    test_sample_cuts: list[str]
    data: Data
    seed: int
    draw_distribution: int | None
    test_subfolder: str
    model_dir: str
    batch_size: int
    take: int
