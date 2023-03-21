from dataclasses import dataclass
from typing import List, Optional, Literal


@dataclass
class Model:
    train_input = Literal['highlevel',
                          'highlevel_constituents',
                          'constituents',
                          'relative_constituents',
                          'interaction_constituents',
                          'deepset_constituents']
    activation: Literal['relu', 'gelu', 'tanh', 'swish']


@dataclass
class BasicFC(Model):
    layer_size: int    # Hidden layer sizes.
    num_layers: int    # Number of highway layers.
    dropout: Optional[float]    # Dropout after FC layers.


@dataclass
class Highway(Model):
    layer_size: int    # Hidden layer sizes.
    num_layers: int    # Number of highway layers.
    dropout: Optional[float]    # Dropout after FC layers.


@dataclass
class PFN(Model):
    Phi_sizes: List[int]    # Hidden layer sizes.
    F_sizes: List[int]    # Number of highway layers.
    Phi_backbone: Literal["cnn", "fc"]
    batch_norm: bool
    Phi_dropout: Optional[float]    # Dropout after FC layers.
    F_dropout: Optional[float]    # Dropout after FC layers.


@dataclass
class EFN(Model):
    Phi_sizes: List[int]    # Hidden layer sizes.
    F_sizes: List[int]    # Number of highway layers.
    Phi_backbone: Literal["cnn", "fc"]
    batch_norm: bool
    Phi_dropout: Optional[float]    # Dropout after FC layers.
    F_dropout: Optional[float]    # Dropout after FC layers.


@dataclass
class BDT(Model):
    num_trees: int
    growing_strategy: str
    max_depth: int
    split_axis: str
    shrinkage: int
    min_examples: int
    num_threads: int
    l2_regularization: float
    max_num_nodes: int
    tmp_dir: str


@dataclass
class Transformer(Model):
    warmup_steps: int  # Number of steps to warmup for
    dropout: float  # Dropout after FFN layer.
    expansion: int  # 4,  number of hidden units in FFN is expansion * embed_dim
    heads: int  # 12, must divide embed_dim
    layers: int  # 6,12
    embed_dim: int
    num_embed_layers: int


@dataclass
class DeParT(Model):
    warmup_steps: int
    embed_dim: int
    num_embed_layers: int
    expansion: int  # 4,  number of hidden units in FFN is expansion * embed_dim
    heads: int  # 12, must divide embed_dim
    layers: int  # 6,12
    class_layers: int
    dropout: float  # Dropout after FFN layer.
    class_dropout: float
    layer_scale_init_value: float
    stochastic_depth_drop_rate: float
    class_stochastic_depth_drop_rate: float
    relative: bool
    interaction_embedding_num_layers: int
    interaction_embedding_layer_size: int


@dataclass
class ParT(Model):
    warmup_steps: int  # Number of steps to warmup for
    particle_block_dropout: float  # Dropout after FFN layer.
    expansion: int  # 4,  number of hidden units in FFN is expansion * embed_dim
    heads: int  # 12, must divide embed_dim
    particle_block_layers: int  # 6,12
    class_block_layers: int
    embed_dim: int
    num_embed_layers: int
    interaction_embedding_num_layers: int
    interaction_embedding_layer_size: int
