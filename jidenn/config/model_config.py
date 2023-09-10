"""
Model configurations. Each model is a subclass of the `Model` dataclass.
For more information about individual models, see the `jidenn.models` module.
"""
from dataclasses import dataclass
from typing import List, Optional, Literal


@dataclass
class Model:
    """Base class for model configurations.
    Args:
        train_input (str): The input to the model. One of (see `jidenn.data.TrainInput` for more details):

            - 'highlevel': The predefined high-level features.
            - 'highlevel_constituents': The high-level features constructed with the constituents.
            - 'constituents': The constituents variables.
            - 'relative_constituents': The constituents variables with only relative variables wrt. jet.
            - 'interaction_constituents': The constituents variables and interaction variables.

        activation (str): The activation function to use. Option: 'relu', 'gelu', 'tanh', 'swish'.
    """
    train_input = Literal['highlevel',
                          'highlevel_constituents',
                          'constituents',
                          'relative_constituents',
                          'interaction_constituents']
    activation: Literal['relu', 'gelu', 'tanh', 'swish']


@dataclass
class FC(Model):
    """Basic fully-connected model.

    Args:
        layer_size (int): Hidden layer size (all are the same).
        num_layers (int): Number of hidden layers.
        dropout (float): Dropout after FC layers.
    """
    layer_size: int    # Hidden layer sizes.
    num_layers: int    # Number of highway layers.
    dropout: Optional[float]    # Dropout after FC layers.


@dataclass
class Highway(Model):
    """Extended fully-connected model with highway connections.

    Args:
        layer_size (int): Hidden layer size (all are the same).
        num_layers (int): Number of hidden layers.
        dropout (float): Dropout after FC layers.

    """
    layer_size: int    # Hidden layer sizes.
    num_layers: int    # Number of highway layers.
    dropout: Optional[float]    # Dropout after FC layers.


@dataclass
class PFN(Model):
    """Particle Flow Network.

    Args:
        Phi_sizes (List[int]): Per-particle mapping hidden layer sizes.
        F_sizes (List[int]): Mapping of the summed per-particle features hidden layer sizes.
        batch_norm (bool): Whether to use batch normalization before all layers.
        Phi_dropout (float): Dropout after Phi layers.
        F_dropout (float): Dropout after F layers.
    """
    Phi_sizes: List[int]    # Hidden layer sizes.
    F_sizes: List[int]    # Number of highway layers.
    Phi_backbone: Literal["cnn", "fc"]
    batch_norm: bool
    Phi_dropout: Optional[float]    # Dropout after FC layers.
    F_dropout: Optional[float]    # Dropout after FC layers.


@dataclass
class EFN(Model):
    """Energy Flow Network.

    Args:
        Phi_sizes (List[int]): Per-particle mapping hidden layer sizes.
        F_sizes (List[int]): Mapping of the summed per-particle features hidden layer sizes.
        batch_norm (bool): Whether to use batch normalization before all layers.
        Phi_dropout (float): Dropout after Phi layers.
        F_dropout (float): Dropout after F layers.
    """
    Phi_sizes: List[int]    # Hidden layer sizes.
    F_sizes: List[int]    # Number of highway layers.
    Phi_backbone: Literal["cnn", "fc"]
    batch_norm: bool
    Phi_dropout: Optional[float]    # Dropout after FC layers.
    F_dropout: Optional[float]    # Dropout after FC layers.


@dataclass
class BDT(Model):
    """Boosted Decision Tree.

    Args:
        num_trees (int): Number of trees.
        growing_strategy (str): Growing strategy. 
        max_depth (int): Maximum depth of the tree.
        split_axis (str): Split axis.
        shrinkage (int): Shrinkage.
        min_examples (int): Minimum number of examples.
        num_threads (int): Number of threads.
        l2_regularization (float): L2 regularization.
        max_num_nodes (int): Maximum number of nodes.
        tmp_dir (str): Temporary directory for checkpointing.
    """
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
    """Transformer model.

    Args:
        embed_dim (int): Embedding dimension, dimension of the internal representation.
        self_attn_layers (int): Number of SelfAttention layers.
        expansion (int): Number of hidden units in FFN is expansion * embed_dim.
        dropout (float): Dropout after FFN layer.
        num_embed_layers (int): Number of embedding layers (with hidden size `embed_dim`)
        heads (int): Number of heads, must be a divisor of `embed_dim`.
    """
    embed_dim: int
    self_attn_layers: int
    heads: int
    expansion: int
    dropout: float
    embed_layers: int


@dataclass
class DeParT(Model):
    """Dynamically Ehnanced Particle Transformer model.

    Args:
        self_attn_layers (int): Number of SelfAttention layers.
        embed_dim (int): Embedding dimension, dimension of the internal representation.
        embed_layers (int): Number of embedding layers (with hidden size `embed_dim`)
        expansion (int): Number of hidden units in FFN is expansion * embed_dim.
        heads (int): Number of heads, must be a divisor of `embed_dim`.
        class_attn_layers (int): Number of ClassAttention layers.
        dropout (float): Dropout in SelfAttention layers.
        class_dropout (float): Dropout in ClassAttention layers.
        layer_scale_init_value (float): Initial value for layer scale.
        stochastic_depth_drop_rate (float): Stochastic depth drop rate.
        class_stochastic_depth_drop_rate (float): Stochastic depth drop rate for ClassAttention layers.
        interaction_embedding_layers (int): Number of interaction embedding layers.
        interaction_embedding_layer_size (int): Size of interaction embedding layers.
    """
    self_attn_layers: int  # 6,12
    embed_dim: int
    embed_layers: int
    expansion: int  # 4,  number of hidden units in FFN is expansion * embed_dim
    heads: int  # 12, must divide embed_dim
    class_attn_layers: int
    dropout: float  # Dropout after FFN layer.
    class_dropout: float
    layer_scale_init_value: float
    stochastic_depth_drop_rate: float
    class_stochastic_depth_drop_rate: float
    interaction_embedding_layers: int
    interaction_embedding_layer_size: int


@dataclass
class ParT(Model):
    """Particle Transformer model.

    Args:
        self_attn_layers (int): Number of SelfAttention layers.
        embed_dim (int): Embedding dimension, dimension of the internal representation.
        embed_layers (int): Number of embedding layers (with hidden size `embed_dim`)
        dropout (float): Dropout in SelfAttention layers.
        expansion (int): Number of hidden units in FFN is expansion * embed_dim.
        heads (int): Number of heads, must be a divisor of `embed_dim`.
        class_attn_layers (int): Number of ClassAttention layers.
        interaction_embedding_layers (int): Number of interaction embedding layers.
        interaction_embedding_layer_size (int): Size of interaction embedding layers.
    """
    self_attn_layers: int  # 6,12
    embed_dim: int
    embed_layers: int
    dropout: float  # Dropout after FFN layer.
    expansion: int  # 4,  number of hidden units in FFN is expansion * embed_dim
    heads: int  # 12, must divide embed_dim
    class_attn_layers: int
    interaction_embedding_layers: int
    interaction_embedding_layer_size: int


@dataclass
class ParticleNet(Model):

    pooling: Literal['average', 'max']
    fc_layers: List[int]
    fc_dropout: List[float]
    edge_knn: List[int]
    edge_layers: List[List[int]]
