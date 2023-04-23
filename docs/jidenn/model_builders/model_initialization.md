Module jidenn.model_builders.model_initialization
=================================================
Module for initializing models using the configuration file.
The config options are explicitly passed to the model classes `__init__` method.
For each model, a function is defined that returns an instance of the model class.
The mapping from string names of the models in the config file is then mapped to the corresponding function.

Functions
---------

    
`get_FC_model(input_size: int, output_layer: keras.engine.base_layer.Layer, args_model: jidenn.config.model_config.FC, preprocess: Optional[keras.engine.base_layer.Layer] = None) ‑> jidenn.models.FC.FCModel`
:   Get an instance of the fully-connected model.
    
    Args:
        input_size (int): Input size of the model.
        output_layer (tf.keras.layers.Layer): Output layer of the model.
        args_model (jidenn.config.model_config.FC): Model configuration.
        preprocess (Optional[tf.keras.layers.Layer], optional): Preprocessing layer. Defaults to None.
    
    Returns:
        FCModel: Fully-connected model.

    
`get_activation(activation: Literal['relu', 'gelu', 'tanh', 'swish']) ‑> Callable[[tensorflow.python.framework.ops.Tensor], tensorflow.python.framework.ops.Tensor]`
:   Map string names of activation functions to the corresponding function.
    
    Args:
        activation (str): Name of the activation function. One of ['relu', 'gelu', 'tanh', 'swish'].
    Returns:
        Callable[[tf.Tensor], tf.Tensor]: Activation function.

    
`get_bdt_model(input_size: int, output_layer: keras.engine.base_layer.Layer, args_model: jidenn.config.model_config.BDT, preprocess: Optional[keras.engine.base_layer.Layer] = None)`
:   Get an instance of the BDT model.
    
    Args:
        args_model (model_config.BDT): Model configuration.
        input_size (int): Input size of the model. UNUSED
        output_layer (tf.keras.layers.Layer): Output layer of the model. UNUSED
        preprocess (Optional[tf.keras.layers.Layer], optional): Preprocessing layer. Defaults to None. UNUSED
    
    Returns:
        tfdf.keras.GradientBoostedTreesModel: BDT model.

    
`get_depart_model(input_size: Union[Tuple[None, int], Tuple[Tuple[None, int], Tuple[None, None, int]]], output_layer: keras.engine.base_layer.Layer, args_model: jidenn.config.model_config.DeParT, preprocess: Optional[keras.engine.base_layer.Layer] = None) ‑> jidenn.models.DeParT.DeParTModel`
:   Get an instance of the DeParT model.
    
    Args:
        input_size (Union[Tuple[None, int], Tuple[Tuple[None, int], Tuple[None, None, int]]]): Input size of the model.
        output_layer (tf.keras.layers.Layer): Output layer of the model.
        args_model (model_config.DeParT): Model configuration.
        preprocess (Optional[tf.keras.layers.Layer], optional): Preprocessing layer. Defaults to None.
    
    Returns:
        DeParTModel: DeParT model.

    
`get_efn_model(input_size: Tuple[None, int], output_layer: keras.engine.base_layer.Layer, args_model: jidenn.config.model_config.EFN, preprocess: Optional[keras.engine.base_layer.Layer] = None) ‑> jidenn.models.EFN.EFNModel`
:   Get an instance of the EFN model.
    
    Args:
        input_size (Tuple[None, int]): Input size of the model.
        output_layer (tf.keras.layers.Layer): Output layer of the model.
        args_model (model_config.EFN): Model configuration.
        preprocess (Optional[tf.keras.layers.Layer], optional): Preprocessing layer. Defaults to None.  
    
    Returns:
        EFNModel: EFN model.

    
`get_highway_model(input_size: int, output_layer: keras.engine.base_layer.Layer, args_model: jidenn.config.model_config.Highway, preprocess: Optional[keras.engine.base_layer.Layer] = None) ‑> jidenn.models.Highway.HighwayModel`
:   Get an instance of the highway model.
    
    Args:
        input_size (int): Input size of the model.
        output_layer (tf.keras.layers.Layer): Output layer of the model.
        args_model (jidenn.config.model_config.Highway): Model configuration.
        preprocess (Optional[tf.keras.layers.Layer], optional): Preprocessing layer. Defaults to None.
    
    Returns:
        HighwayModel: Highway model.

    
`get_part_model(input_size: Union[Tuple[None, int], Tuple[Tuple[None, int], Tuple[None, None, int]]], output_layer: keras.engine.base_layer.Layer, args_model: jidenn.config.model_config.ParT, preprocess: Optional[keras.engine.base_layer.Layer] = None) ‑> jidenn.models.ParT.ParTModel`
:   Get an instance of the ParT model.
    
    Args:
        input_size (Union[Tuple[None, int], Tuple[Tuple[None, int], Tuple[None, None, int]]]): Input size of the model.
        output_layer (tf.keras.layers.Layer): Output layer of the model.
        args_model (model_config.ParT): Model configuration.
        preprocess (Optional[tf.keras.layers.Layer], optional): Preprocessing layer. Defaults to None.
    
    Returns:
        ParTModel: ParT model.

    
`get_pfn_model(input_size: Tuple[None, int], output_layer: keras.engine.base_layer.Layer, args_model: jidenn.config.model_config.PFN, preprocess: Optional[keras.engine.base_layer.Layer] = None) ‑> jidenn.models.PFN.PFNModel`
:   Get an instance of the PFN model.
    
    Args:
        input_size (Tuple[None, int]): Input size of the model.
        output_layer (tf.keras.layers.Layer): Output layer of the model.
        args_model (model_config.PFN): Model configuration.
        preprocess (Optional[tf.keras.layers.Layer], optional): Preprocessing layer. Defaults to None.
    
    Returns:
        PFNModel: PFN model.

    
`get_transformer_model(input_size: Tuple[None, int], output_layer: keras.engine.base_layer.Layer, args_model: jidenn.config.model_config.Transformer, preprocess: Optional[keras.engine.base_layer.Layer] = None) ‑> jidenn.models.Transformer.TransformerModel`
:   Get an instance of the transformer model.
    
    Args:
        input_size (Tuple[None, int]): Input size of the model.
        output_layer (tf.keras.layers.Layer): Output layer of the model.
        args_model (model_config.Transformer): Model configuration.
        preprocess (Optional[tf.keras.layers.Layer], optional): Preprocessing layer. Defaults to None.  
    
    Returns:
        TransformerModel: Transformer model.

    
`model_getter_lookup(model_name: Literal['fc', 'highway', 'pfn', 'efn', 'transformer', 'part', 'depart', 'bdt']) ‑> Callable[[Any, keras.engine.base_layer.Layer, jidenn.config.model_config.Model, Optional[keras.engine.base_layer.Layer]], keras.engine.training.Model]`
:   Get a model getter function.
    
    Args:
        model_name (str): Name of the model. Options are 'fc', 'highway', 'pfn', 'efn', 'transformer', 'part', 'depart', 'bdt'.
        
    Returns:
        Callable[[Any, tf.keras.layers.Layer, model_config.Model, Optional[tf.keras.layers.Layer]], tf.keras.Model]: Model getter function.