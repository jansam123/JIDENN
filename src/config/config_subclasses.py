from dataclasses import dataclass
from typing import List, Union, Optional, Literal

from .model_config import BasicFC, Highway, BDT, Transformer, DeParT, ParT, PFN


@dataclass
class Variables:
    perJet: List[str]
    perJetTuple: List[str]
    perEvent: List[str]


@dataclass
class Data:
    labels: List[str]    # list of labels to use.
    num_labels: int   # Number of labels to use.
    input_size: int    # Number of input features.
    target: str
    variables: Variables
    weight: Union[str, None]
    cut: Union[str, None]
    tttree_name: str  # Name of the tree in the root file.
    gluon: int
    quark: int
    raw_gluon: int
    raw_quarks: List[int]
    raw_unknown: List[int]
    path: str   # Path to data folder containing folder of *.root files.
    JZ_slices: Union[List[int], None]   # Slices of JZ to use.
    JZ_cut: Union[List[str], None]   # Cut to apply to JZ slices.
    JZ_weights: Union[List[float], None]  # Weights to apply to JZ slices.
    cached: Union[str, None]   # Path to cached data.


@dataclass
class Dataset:
    batch_size: int   # Batch size.
    validation_step: int   # Validation every n batches.
    dev_size: float   # Size of dev dataset.
    test_size: float  # Size of test dataset.
    take: Union[int, None]   # Length of data to use.
    shuffle_buffer: Union[int, None]   # Size of shuffler buffer.
    epochs: int


@dataclass
class Params:
    model: str  # Model to use, options: 'basic_fc', 'transformer'.
    epochs: int  # Number of epochs.
    seed: int   # Random seed.
    threads: int   # Maximum number of threads to use.
    debug: bool   # Debug mode.
    logdir: str   # Path to log directory.
    checkpoint: Optional[str]   # Make checkpoint.
    backup: Optional[str]   # Backup model.
    load_checkpoint_path: Union[str, None]   # Path to checkpoint to load.


@dataclass
class Preprocess:
    normalize: bool  # Normalize data.
    draw_distribution: Union[int, None]   # Number of events to draw distribution for.
    normalization_size: Union[int, None]  # Size of normalization dataset.
    min_max_path: Union[str, None]  # Path to min max values.


@dataclass
class Optimizer:
    name: Literal['LAMB', 'Adam']
    learning_rate: float
    label_smoothing: float
    decay_steps: Optional[int]
    warmup_steps: int
    beta_1: float
    beta_2: float
    epsilon: float
    clipnorm: Optional[float]
    weight_decay: float


@dataclass
class Models:
    basic_fc: BasicFC
    transformer: Transformer
    bdt: BDT
    highway: Highway
    part: ParT
    depart: DeParT
    pfn: PFN
