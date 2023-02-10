from dataclasses import dataclass
from typing import List, Union, Optional, Literal

@dataclass
class BasicFC:
    layer_size: int    # Hidden layer sizes.
    num_layers: int    # Number of highway layers.
    dropout: Union[float, None]    # Dropout after FC layers.
    activation: str


@dataclass
class Highway:
    layer_size: int    # Hidden layer sizes.
    num_layers: int    # Number of highway layers.
    dropout: Union[float, None]    # Dropout after FC layers.
    activation: str


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
    activation: str
    
@dataclass
class Transformer:
    warmup_steps: int  # Number of steps to warmup for
    dropout: float  # Dropout after FFN layer.
    expansion: int  # 4,  number of hidden units in FFN is expansion * embed_dim
    heads: int  # 12, must divide embed_dim
    layers: int  # 6,12
    embed_dim: int
    num_embed_layers: int
    activation: str


@dataclass
class DeParT:
    warmup_steps: int
    embed_dim: int
    num_embed_layers: int
    expansion: int  # 4,  number of hidden units in FFN is expansion * embed_dim
    heads: int  # 12, must divide embed_dim
    layers: int  # 6,12
    class_layers: int
    dropout: float  # Dropout after FFN layer.
    layer_scale_init_value: float
    stochastic_depth_drop_rate: float
    interaction: bool
    activation: str


@dataclass
class ParT:
    warmup_steps: int  # Number of steps to warmup for
    particle_block_dropout: float  # Dropout after FFN layer.
    expansion: int  # 4,  number of hidden units in FFN is expansion * embed_dim
    heads: int  # 12, must divide embed_dim
    particle_block_layers: int  # 6,12
    class_block_layers: int
    embed_dim: int
    num_embed_layers: int
    interaction: bool
    interaction_embedding_num_layers: int
    interaction_embedding_layer_size: int
    activation: str


    