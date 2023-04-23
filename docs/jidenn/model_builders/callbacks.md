Module jidenn.model_builders.callbacks
======================================
Module containing the custom callbacks and a function that returns 
a list of callbacks to be used during the training process.

Functions
---------

    
`get_callbacks(base_logdir: str, epochs: int, log: logging.Logger, checkpoint: Optional[str] = 'checkpoints', backup: Optional[str] = 'backup') ‑> List[keras.callbacks.Callback]`
:   Returns a list of Keras callbacks for a training session.
    
    Args:
        base_log_dir (str): The base directory to use for saving logs, checkpoints, and backups.
        epochs (int): The number of epochs to train for.
        log (logging.Logger): A logger object to use for logging information during training.
        checkpoint (str): The directory to use for saving model checkpoints. If None, no checkpoints will be saved.
        backup (str): The directory to use for saving backups of the training session. If None, no backups will be saved.
    
    Returns:
        A list of Keras callbacks to use during training. The list contains a `tf.keras.callbacks.TensorBoard` callback for logging training information, a `jidenn.callbacks.LogCallback.LogCallback` callback for logging training information, a `jidenn.callbacks.BestNModelCheckpoint.BestNModelCheckpoint` callback for saving model checkpoints, and a `tf.keras.callbacks.BackupAndRestore` callback for saving backups of the training session.

Classes
-------

`BestNModelCheckpoint(filepath: str, max_to_keep: int, keep_most_recent: bool = True, monitor: str = 'val_loss', mode: Literal['min', 'max'] = 'min', **kwargs)`
:   Custom ModelCheckpoint Callback that saves the best N checkpoints based on a specified monitor metric.
    
    Original source: https://github.com/schustmi/tf_utils/blob/915fe5e231ca302b28cd02dc8ac2e4c772a62e0b/tf_utils/callbacks.py#L34
    
    This callback is a modification of the `ModelCheckpoint` callback in TensorFlow Keras, allowing for keeping
    the best N checkpoints based on a specified monitor metric. The checkpoints are saved in a directory specified
    by `filepath` argument, and the N best checkpoints are kept based on the value of the monitor metric. Optionally,
    the most recent checkpoint can also be kept.
    
    Args:
        filepath (str): path where the checkpoints will be saved. Make sure
            to pass a format string as otherwise the checkpoint will be overridden each step.
            (see `tf.keras.callbacks.ModelCheckpoint` for detailed formatting options)
        max_to_keep (int): Maximum number of best checkpoints to keep.
        keep_most_recent (bool): if True, the most recent checkpoint will be saved in addition to
            the best `max_to_keep` ones.
        monitor (str): Name of the metric to monitor. The value for this metric will be used to
            decide which checkpoints will be kept. See `keras.callbacks.ModelCheckpoint` for
            more information.
        mode (str): Depending on `mode`, the checkpoints with the highest ('max') 
            or lowest ('min') values in the monitored quantity will be kept.
        **kwargs: Additional keyword arguments to pass to the parent class `tf.keras.callbacks.ModelCheckpoint`.

    ### Ancestors (in MRO)

    * keras.callbacks.ModelCheckpoint
    * keras.callbacks.Callback

`LogCallback(epochs: int, logger: logging.Logger)`
:   Callback to log the training progress to a specified logging.Logger object. Logs the total time elapsed during the epoch (ETA).
    
    Args:
        epochs (int): The total number of epochs.
        logger (logging.Logger): The logging object to use.
    
    Example:
        An example output of the logger:
    
        ```
        Epoch 1/5: ETA: 01:18:43 - loss: 0.5569 - accuracy: 0.7089 - val_loss: 0.533 - val_accuracy: 0.7307
        Epoch 2/5: ETA: 01:15:50 - loss: 0.5333 - accuracy: 0.7289 - val_loss: 0.5291 - val_accuracy: 0.7328
        Epoch 3/5: ETA: 01:15:12 - loss: 0.5304 - accuracy: 0.7311 - val_loss: 0.5266 - val_accuracy: 0.7344
        Epoch 4/5: ETA: 01:15:18 - loss: 0.5281 - accuracy: 0.7328 - val_loss: 0.525 - val_accuracy: 0.7351
        Epoch 5/5: ETA: 01:14:56 - loss: 0.5262 - accuracy: 0.7342 - val_loss: 0.5231 - val_accuracy: 0.7374
        ```

    ### Ancestors (in MRO)

    * keras.callbacks.Callback

    ### Methods

    `on_epoch_begin(self, epoch: int, logs: Optional[dict] = None) ‑> None`
    :   Called at the beginning of every epoch. Sets the epoch start time.
        
        Args:
            epoch (int): Index of epoch.
            logs (dict, optional): Dictionary of logs for the current epoch.
        
        Returns:
            None

    `on_epoch_end(self, epoch: int, logs: dict) ‑> None`
    :   Called at the end of every epoch. Logs the epoch progress and calculates the total time elapsed during the epoch (ETA).
        
        Args:
            epoch (int): Index of epoch.
            logs (dict): Dictionary of logs for the current epoch.
        
        Returns:
            None