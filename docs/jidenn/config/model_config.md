Module jidenn.config.model_config
=================================
Model configurations. Each model is a subclass of the `Model` dataclass.
For more information about individual models, see the `jidenn.models` module.

Classes
-------

`BDT(activation: Literal['relu', 'gelu', 'tanh', 'swish'], num_trees: int, growing_strategy: str, max_depth: int, split_axis: str, shrinkage: int, min_examples: int, num_threads: int, l2_regularization: float, max_num_nodes: int, tmp_dir: str)`
:   Boosted Decision Tree.
    
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

    ### Ancestors (in MRO)

    * jidenn.config.model_config.Model

    ### Class variables

    `growing_strategy: str`
    :

    `l2_regularization: float`
    :

    `max_depth: int`
    :

    `max_num_nodes: int`
    :

    `min_examples: int`
    :

    `num_threads: int`
    :

    `num_trees: int`
    :

    `shrinkage: int`
    :

    `split_axis: str`
    :

    `tmp_dir: str`
    :

`DeParT(activation: Literal['relu', 'gelu', 'tanh', 'swish'], self_attn_layers: int, embed_dim: int, embed_layers: int, expansion: int, heads: int, class_attn_layers: int, dropout: float, class_dropout: float, layer_scale_init_value: float, stochastic_depth_drop_rate: float, class_stochastic_depth_drop_rate: float, interaction_embedding_layers: int, interaction_embedding_layer_size: int)`
:   Dynamically Ehnanced Particle Transformer model.
    
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

    ### Ancestors (in MRO)

    * jidenn.config.model_config.Model

    ### Class variables

    `class_attn_layers: int`
    :

    `class_dropout: float`
    :

    `class_stochastic_depth_drop_rate: float`
    :

    `dropout: float`
    :

    `embed_dim: int`
    :

    `embed_layers: int`
    :

    `expansion: int`
    :

    `heads: int`
    :

    `interaction_embedding_layer_size: int`
    :

    `interaction_embedding_layers: int`
    :

    `layer_scale_init_value: float`
    :

    `self_attn_layers: int`
    :

    `stochastic_depth_drop_rate: float`
    :

`EFN(activation: Literal['relu', 'gelu', 'tanh', 'swish'], Phi_sizes: List[int], F_sizes: List[int], Phi_backbone: Literal['cnn', 'fc'], batch_norm: bool, Phi_dropout: Optional[float], F_dropout: Optional[float])`
:   Energy Flow Network.
    
    Args:
        Phi_sizes (List[int]): Per-particle mapping hidden layer sizes.
        F_sizes (List[int]): Mapping of the summed per-particle features hidden layer sizes.
        batch_norm (bool): Whether to use batch normalization before all layers.
        Phi_dropout (float): Dropout after Phi layers.
        F_dropout (float): Dropout after F layers.

    ### Ancestors (in MRO)

    * jidenn.config.model_config.Model

    ### Class variables

    `F_dropout: Optional[float]`
    :

    `F_sizes: List[int]`
    :

    `Phi_backbone: Literal['cnn', 'fc']`
    :

    `Phi_dropout: Optional[float]`
    :

    `Phi_sizes: List[int]`
    :

    `batch_norm: bool`
    :

`FC(activation: Literal['relu', 'gelu', 'tanh', 'swish'], layer_size: int, num_layers: int, dropout: Optional[float])`
:   Basic fully-connected model.
    
    Args:
        layer_size (int): Hidden layer size (all are the same).
        num_layers (int): Number of hidden layers.
        dropout (float): Dropout after FC layers.

    ### Ancestors (in MRO)

    * jidenn.config.model_config.Model

    ### Class variables

    `dropout: Optional[float]`
    :

    `layer_size: int`
    :

    `num_layers: int`
    :

`Highway(activation: Literal['relu', 'gelu', 'tanh', 'swish'], layer_size: int, num_layers: int, dropout: Optional[float])`
:   Extended fully-connected model with highway connections.
    
    Args:
        layer_size (int): Hidden layer size (all are the same).
        num_layers (int): Number of hidden layers.
        dropout (float): Dropout after FC layers.

    ### Ancestors (in MRO)

    * jidenn.config.model_config.Model

    ### Class variables

    `dropout: Optional[float]`
    :

    `layer_size: int`
    :

    `num_layers: int`
    :

`Model(activation: Literal['relu', 'gelu', 'tanh', 'swish'])`
:   Base class for model configurations.
    Args:
        train_input (str): The input to the model. One of (see `jidenn.data.TrainInput` for more details):
    
            - 'highlevel': The predefined high-level features.
            - 'highlevel_constituents': The high-level features constructed with the constituents.
            - 'constituents': The constituents variables.
            - 'relative_constituents': The constituents variables with only relative variables wrt. jet.
            - 'interaction_constituents': The constituents variables and interaction variables.
    
        activation (str): The activation function to use. Option: 'relu', 'gelu', 'tanh', 'swish'.

    ### Descendants

    * jidenn.config.model_config.BDT
    * jidenn.config.model_config.DeParT
    * jidenn.config.model_config.EFN
    * jidenn.config.model_config.FC
    * jidenn.config.model_config.Highway
    * jidenn.config.model_config.PFN
    * jidenn.config.model_config.ParT
    * jidenn.config.model_config.Transformer

    ### Class variables

    `activation: Literal['relu', 'gelu', 'tanh', 'swish']`
    :

    `train_input`
    :

`PFN(activation: Literal['relu', 'gelu', 'tanh', 'swish'], Phi_sizes: List[int], F_sizes: List[int], Phi_backbone: Literal['cnn', 'fc'], batch_norm: bool, Phi_dropout: Optional[float], F_dropout: Optional[float])`
:   Particle Flow Network.
    
    Args:
        Phi_sizes (List[int]): Per-particle mapping hidden layer sizes.
        F_sizes (List[int]): Mapping of the summed per-particle features hidden layer sizes.
        batch_norm (bool): Whether to use batch normalization before all layers.
        Phi_dropout (float): Dropout after Phi layers.
        F_dropout (float): Dropout after F layers.

    ### Ancestors (in MRO)

    * jidenn.config.model_config.Model

    ### Class variables

    `F_dropout: Optional[float]`
    :

    `F_sizes: List[int]`
    :

    `Phi_backbone: Literal['cnn', 'fc']`
    :

    `Phi_dropout: Optional[float]`
    :

    `Phi_sizes: List[int]`
    :

    `batch_norm: bool`
    :

`ParT(activation: Literal['relu', 'gelu', 'tanh', 'swish'], self_attn_layers: int, embed_dim: int, embed_layers: int, dropout: float, expansion: int, heads: int, class_attn_layers: int, interaction_embedding_layers: int, interaction_embedding_layer_size: int)`
:   Particle Transformer model.
    
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

    ### Ancestors (in MRO)

    * jidenn.config.model_config.Model

    ### Class variables

    `class_attn_layers: int`
    :

    `dropout: float`
    :

    `embed_dim: int`
    :

    `embed_layers: int`
    :

    `expansion: int`
    :

    `heads: int`
    :

    `interaction_embedding_layer_size: int`
    :

    `interaction_embedding_layers: int`
    :

    `self_attn_layers: int`
    :

`Transformer(activation: Literal['relu', 'gelu', 'tanh', 'swish'], embed_dim: int, self_attn_layers: int, heads: int, expansion: int, dropout: float, embed_layers: int)`
:   Transformer model.
    
    Args:
        embed_dim (int): Embedding dimension, dimension of the internal representation.
        self_attn_layers (int): Number of SelfAttention layers.
        expansion (int): Number of hidden units in FFN is expansion * embed_dim.
        dropout (float): Dropout after FFN layer.
        num_embed_layers (int): Number of embedding layers (with hidden size `embed_dim`)
        heads (int): Number of heads, must be a divisor of `embed_dim`.

    ### Ancestors (in MRO)

    * jidenn.config.model_config.Model

    ### Class variables

    `dropout: float`
    :

    `embed_dim: int`
    :

    `embed_layers: int`
    :

    `expansion: int`
    :

    `heads: int`
    :

    `self_attn_layers: int`
    :