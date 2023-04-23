Module jidenn.model_builders.normalization_initialization
=========================================================
Module for initializing normalization layers and adapting them to the dataset,
i.e. calculating the mean and std of the dataset.

Functions
---------

    
`get_normalization(dataset: tensorflow.python.data.ops.dataset_ops.DatasetV2, log: logging.Logger, adapt: bool = True, ragged: bool = False, normalization_steps: Optional[int] = None, interaction: Optional[bool] = None) ‑> Union[keras.layers.preprocessing.normalization.Normalization, Tuple[keras.layers.preprocessing.normalization.Normalization, keras.layers.preprocessing.normalization.Normalization]]`
:   Function returning normalization layer(s) adapted to the dataset if requested.
    Unadapted normalization layer(s) are asumed to have weights loaded from checkpoint.
    
    Args:
        dataset (tf.data.Dataset): Dataset to adapt the normalization layer to.
        log (logging.Logger): Logger.
        adapt (bool, optional): Whether to adapt the normalization layer to the dataset. Defaults to True.
        ragged (bool, optional): Whether the dataset samples are ragged. Defaults to False.
        normalization_steps (int, optional): Number of batches to use for adaptation. Defaults to None.
        interaction (bool, optional): Whether to adapt the normalization layer to the interaction variables. Defaults to None.
    
    Returns:
        Union[tf.keras.layers.Normalization, Tuple[tf.keras.layers.Normalization, tf.keras.layers.Normalization]]: Return `None` if the model is `bdt` or unknown model,
            one layer if there is no interaction, two layers as a `tuple` if there is interaction.