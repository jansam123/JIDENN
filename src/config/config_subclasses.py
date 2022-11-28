from dataclasses import dataclass


@dataclass
class BasicFC:
    layer_size: int    # Hidden layer sizes.
    num_layers: int    # Number of highway layers.
    dropout: float | None    # Dropout after FC layers.
    rnn_dim: int  # dimesion of RNN cells


@dataclass
class Highway:
    layer_size: int    # Hidden layer sizes.
    num_layers: int    # Number of highway layers.
    dropout: float | None    # Dropout after FC layers.


@dataclass
class BDT:
    num_trees: int
    growing_strategy: str
    max_depth: int
    split_axis: str
    categorical_algorithm: str
    shrinkage: int
    min_examples: int
    num_threads: int
    l2_regularization: float


@dataclass
class Variables:
    perJet: list[str]
    perJetTuple: list[str]
    perEvent: list[str]


@dataclass
class Data:
    labels: list[str]    # list of labels to use.
    num_labels: int   # Number of labels to use.
    input_size: int    # Number of input features.
    target: str
    variables: Variables
    weight: str | None
    cut: str | None
    tttree_name: str  # Name of the tree in the root file.
    gluon: int
    quark: int
    raw_gluon: int
    raw_quarks: list[int]
    raw_unknown: list[int]
    path: str   # Path to data folder containing folder of *.root files.
    JZ_slices: list[int] | None   # Slices of JZ to use.
    JZ_cut: list[str] | None   # Cut to apply to JZ slices.
    JZ_weights: list[float] | None   # Weights to apply to JZ slices.
    cached: str | None   # Path to cached data.


@dataclass
class Dataset:
    batch_size: int   # Batch size.
    validation_step: int   # Validation every n batches.
    dev_size: float   # Size of dev dataset.
    test_size: float  # Size of test dataset.
    take: int | None   # Length of data to use.
    shuffle_buffer: int | None   # Size of shuffler buffer.


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
    # TODO
    decay_steps: int  # Number of steps to decay for.
    weight_decay: float


@dataclass
class Preprocess:
    normalize: bool  # Normalize data.
    draw_distribution: int | None   # Number of events to draw distribution for.
    normalization_size: int | None  # Size of normalization dataset.
    min_max_path: str | None  # Path to min max values.


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
class ParT:
    warmup_steps: int  # Number of steps to warmup for
    particle_block_dropout: float  # Dropout after FFN layer.
    transformer_expansion: int  # 4,  number of hidden units in FFN is transformer_expansion * embed_dim
    transformer_heads: int  # 12, must divide embed_dim
    particle_block_layers: int  # 6,12
    class_block_layers: int
    embed_dim: int
    num_embed_layers: int
