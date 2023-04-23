Module jidenn.model_builders.ModelBuilder
=========================================
Module for building models from config and compiling them.

Classes
-------

`ModelBuilder(model_name: Literal['fc', 'highway', 'pfn', 'efn', 'transformer', 'part', 'depart', 'bdt'], args_model: jidenn.config.config.Models, input_size: Union[int, Tuple[None, int], Tuple[Tuple[None, int], Tuple[None, None, int]]], num_labels: int, args_optimizer: jidenn.config.config.Optimizer, preprocess: Union[keras.engine.base_layer.Layer, ForwardRef(None), Tuple[keras.engine.base_layer.Layer, keras.engine.base_layer.Layer]] = None)`
:   Class for building models from config.
    Provides a facade between pure model initialization and model building with output layer, optimizer, loss, and metrics, with subsequent compilation.
    
    Args:
        model_name (str): Name of model to build. Options are 'fc', 'highway', 'pfn', 'efn', 'transformer', 'part', 'depart', 'bdt'
        args_model (config.Models): Model config
        input_size (Union[int, Tuple[None, int], Tuple[Tuple[None, int], Tuple[None, None, int]]]): Input size
        num_labels (int): Number of labels, i.e. size of output layer
        args_optimizer (config.Optimizer): Optimizer config
        preprocess (Union[tf.keras.layers.Layer, None, Tuple[tf.keras.layers.Layer, tf.keras.layers.Layer]], optional): Preprocessing layer(s). Defaults to None.

    ### Class variables

    `args_model: jidenn.config.config.Models`
    :

    `args_optimizer: jidenn.config.config.Optimizer`
    :

    `input_size: Union[int, Tuple[None, int], Tuple[Tuple[None, int], Tuple[None, None, int]]]`
    :

    `model_name: Literal['fc', 'highway', 'pfn', 'efn', 'transformer', 'part', 'depart', 'bdt']`
    :

    `num_labels: int`
    :

    `preprocess: Union[keras.engine.base_layer.Layer, ForwardRef(None), Tuple[keras.engine.base_layer.Layer, keras.engine.base_layer.Layer]]`
    :

    ### Instance variables

    `compiled_model: keras.engine.training.Model`
    :   Compiled Model.

    `loss: keras.losses.Loss`
    :   Loss function used in training.

    `metrics: List[keras.metrics.base_metric.Metric]`
    :   Metrics used in training.

    `model: keras.engine.training.Model`
    :   Builds model from config.

    `optimizer: keras.optimizers.optimizer_experimental.optimizer.Optimizer`
    :   Instantiates optimizer from config.

    `output_layer: keras.engine.base_layer.Layer`
    :   Output layer for model. It is the same for all models.