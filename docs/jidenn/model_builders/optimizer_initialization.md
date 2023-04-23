Module jidenn.model_builders.optimizer_initialization
=====================================================
Module for initializing the optimizer from the config file.
The corresponding config dataclass is defined in `jidenn.config.config.Optimizer`.

Functions
---------

    
`get_optimizer(args_optimizer: jidenn.config.config.Optimizer) ‑> keras.optimizers.optimizer_experimental.optimizer.Optimizer`
:   Initializes the optimizer from the config file.
    Possible optimizers are `tf.optimizers.Adam` and `tfa.optimizers.LAMB`.
    If the `weight_decay` parameter is set, the `tfa.optimizers.AdamW` optimizer is used.
    
    Args:
        args_optimizer (jidenn.config.config.Optimizer): config dataclass for the optimizer
    
    Raises:
        NotImplementedError: Raised if the optimizer is not supported.
    
    Returns:
        tf.keras.optimizers.Optimizer: Optimizer object with set parameters.