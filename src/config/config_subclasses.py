from dataclasses import dataclass
from typing import List, Union


@dataclass
class BasicFC:
    layer_size: int    # Hidden layer sizes.
    num_layers: int    # Number of highway layers.
    dropout: Union[float, None]    # Dropout after FC layers.
    rnn_dim: int  # dimesion of RNN cells


@dataclass
class Highway:
    layer_size: int    # Hidden layer sizes.
    num_layers: int    # Number of highway layers.
    dropout: Union[float, None]    # Dropout after FC layers.


@dataclass
class BDT:
    num_trees: int
    growing_strategy: str
    max_depth: int
    split_axis: str
    shrinkage: int
    min_examples: int
    num_threads: int
    l2_regularization: float


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


@dataclass
class Params:
    model: str  # Model to use, options: 'basic_fc', 'transformer'.
    epochs: int  # Number of epochs.
    label_smoothing: float  # Label smoothing.
    learning_rate: float
    seed: int   # Random seed.
    threads: int   # Maximum number of threads to use.
    debug: bool   # Debug mode.
    logdir: str   # Path to log directory.
    activation: str   # Activation function to use.
    decay_steps: int  # Number of steps to decay for.
    weight_decay: float
    beta_1: float
    beta_2: float
    epsilon: float
    clip_norm: float


@dataclass
class Preprocess:
    normalize: bool  # Normalize data.
    draw_distribution: Union[int, None]   # Number of events to draw distribution for.
    normalization_size: Union[int, None]  # Size of normalization dataset.
    min_max_path: Union[str, None]  # Path to min max values.


@dataclass
class Transformer:
    warmup_steps: int  # Number of steps to warmup for
    transformer_dropout: float  # Dropout after FFN layer.
    transformer_expansion: int  # 4,  number of hidden units in FFN is transformer_expansion * embed_dim
    transformer_heads: int  # 12, must divide embed_dim
    transformer_layers: int  # 6,12
    embed_dim: int
    num_embed_layers: int


@dataclass
class DeParT:
    warmup_steps: int
    embed_dim: int
    num_embed_layers: int
    expansion: int  # 4,  number of hidden units in FFN is transformer_expansion * embed_dim
    heads: int  # 12, must divide embed_dim
    layers: int  # 6,12
    class_layers: int
    dropout: float  # Dropout after FFN layer.
    layer_scale_init_value: float
    stochastic_depth_drop_rate: float
    interaction: bool


@dataclass
class ParT:
    warmup_steps: int  # Number of steps to warmup for
    particle_block_dropout: float  # Dropout after FFN layer.
    transformer_expansion: int  # 4,  number of hidden units in FFN is transformer_expansion * embed_dim
    transformer_heads: int  # 12, must divide embed_dim
    particle_block_layers: int  # 6,12
    class_block_layers: int
    embed_dim: int
    num_embed_layers: int
    interaction: bool
    interaction_embedding_num_layers: int
    interaction_embedding_layer_size: int
