"""
Module containing the custom callbacks and a function that returns 
a list of callbacks to be used during the training process.
"""

import tensorflow as tf
import os
from logging import Logger
from typing import List, Optional, Literal
from tensorflow.python.keras.utils import tf_utils
import tensorflow as tf
import shutil
import logging
from datetime import datetime


log = logging.getLogger(__name__)


class LogCallback(tf.keras.callbacks.Callback):
    """
    Callback to log the training progress to a specified logging.Logger object. Logs the total time elapsed during the epoch (ETA).

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

    """

    def __init__(self, epochs: int, logger: Logger) -> None:
        self.epoch_start_time = None
        self.specified_logger = logger
        self.total_epochs = epochs

    def on_epoch_begin(self, epoch: int, logs: Optional[dict] = None) -> None:
        """
        Called at the beginning of every epoch. Sets the epoch start time.

        Args:
            epoch (int): Index of epoch.
            logs (dict, optional): Dictionary of logs for the current epoch.

        Returns:
            None
        """
        self.epoch_start_time = datetime.utcnow()

    def on_epoch_end(self, epoch: int, logs: dict) -> None:
        """
        Called at the end of every epoch. Logs the epoch progress and calculates the total time elapsed during the epoch (ETA).

        Args:
            epoch (int): Index of epoch.
            logs (dict): Dictionary of logs for the current epoch.

        Returns:
            None
        """

        eta = datetime.utcnow() - self.epoch_start_time if self.epoch_start_time is not None else '-:--:--'
        eta = datetime.strptime(str(eta), "%H:%M:%S.%f")

        total_steps = self.model.optimizer.iterations.numpy()
        log_str = f"Epoch {epoch+1}/{self.total_epochs}, Total Steps {total_steps}: "
        log_str += f"ETA: {eta:%T} - "
        log_str += " - ".join([f"{k}: {v:.4}" for k, v in logs.items()])
        self.specified_logger.info(log_str)

class AdditionalValidation(tf.keras.callbacks.Callback):

    def __init__(self, dataset: tf.data.Dataset, name: str = 'val2', file_writer = None) -> None:
        super(AdditionalValidation).__init__()
        self.dataset = dataset
        self.val_name = name
        self.file_writer = file_writer

    def on_epoch_end(self, epoch: int, logs: dict) -> None:

        results = self.model.evaluate(self.dataset) 
        
        for k, v in zip(self.model.metrics_names, results):
            logs[f'{self.val_name}_{k}'] = v
            if self.file_writer is not None:
                with self.file_writer.as_default():
                    tf.summary.scalar(f'epoch_{k}', v, step=epoch)
            


class BestNModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    """Custom ModelCheckpoint Callback that saves the best N checkpoints based on a specified monitor metric.

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

    """

    def __init__(self,
                 filepath: str,
                 max_to_keep: int,
                 keep_most_recent: bool = True,
                 monitor: str = 'val_loss',
                 mode: Literal['min', 'max'] = 'min',
                 **kwargs):
        if kwargs.pop('save_best_only', None):
            log.warning("Setting `save_best_only` to False.")

        if max_to_keep < 1:
            log.warning("BestNModelCheckpoint parameter `max_to_keep` must be greater than 0, setting it to 1.")
            max_to_keep = 1

        super().__init__(filepath,
                         monitor=monitor,
                         mode=mode,
                         save_best_only=False,
                         **kwargs)
        self._keep_count = max_to_keep
        self._checkpoints = {}

        self._keep_most_recent = keep_most_recent
        if self._keep_most_recent:
            self._most_recent_checkpoint = None

    def _save_model(self, epoch, batch, logs):
        super()._save_model(epoch, batch, logs)
        logs = tf_utils.sync_to_numpy_or_python_type(logs or {})
        filepath = self._get_file_path(epoch, batch, logs)

        if not self._checkpoint_exists(filepath):
            # Did not save a checkpoint for current epoch
            return

        value = logs.get(self.monitor)

        if self._keep_most_recent:
            # delay adding to list of current checkpoints until next save
            # if we should always keep the most recent checkpoint
            if self._most_recent_checkpoint:
                self._checkpoints.update(self._most_recent_checkpoint)
            self._most_recent_checkpoint = {filepath: value}
        else:
            self._checkpoints[filepath] = value

        if len(self._checkpoints) > self._keep_count:
            self._delete_worst_checkpoint()

    def _delete_worst_checkpoint(self):
        worst_checkpoint = None  # tuple (filepath, value)

        for checkpoint in self._checkpoints.items():
            if worst_checkpoint is None or self.monitor_op(worst_checkpoint[1], checkpoint[1]):
                worst_checkpoint = checkpoint

        self._checkpoints.pop(worst_checkpoint[0])
        self._delete_checkpoint_files(worst_checkpoint[0])

    @staticmethod
    def _delete_checkpoint_files(checkpoint_path):
        log.info(f"Removing files for checkpoint '{checkpoint_path}'")

        if os.path.isdir(checkpoint_path):
            # SavedModel format delete the whole directory
            shutil.rmtree(checkpoint_path)
            return

        for f in tf.io.gfile.glob(checkpoint_path + '*'):
            os.remove(f)


def get_callbacks(base_logdir: str,
                  epochs: int,
                  log: Logger,
                  checkpoint: Optional[str] = None,
                  backup: Optional[str] = 'backup',
                  backup_freq: Optional[int] = None,
                  additional_val_dataset: Optional[tf.data.Dataset] = None,
                  additional_val_name: Optional[str] = 'val2',) -> List[tf.keras.callbacks.Callback]:
    """
    Returns a list of Keras callbacks for a training session.

    Args:
        base_log_dir (str): The base directory to use for saving logs, checkpoints, and backups.
        epochs (int): The number of epochs to train for.
        log (logging.Logger): A logger object to use for logging information during training.
        checkpoint (str): The directory to use for saving model checkpoints. If None, no checkpoints will be saved.
        backup (str): The directory to use for saving backups of the training session. If None, no backups will be saved.
        backup_freq (int, optional): The frequency (in batches) at which to save backups of the training session. If None, backups will only be saved at the end of each epoch.

    Returns:
        A list of Keras callbacks to use during training. The list contains a `tf.keras.callbacks.TensorBoard` callback for logging training information, a `jidenn.callbacks.LogCallback.LogCallback` callback for logging training information, a `jidenn.callbacks.BestNModelCheckpoint.BestNModelCheckpoint` callback for saving model checkpoints, and a `tf.keras.callbacks.BackupAndRestore` callback for saving backups of the training session.
    """
    callbacks = []
    if additional_val_dataset is not None and additional_val_name is not None:
        file_writer = tf.summary.create_file_writer(os.path.join(base_logdir, 'test'))
        callbacks.append(AdditionalValidation(additional_val_dataset, name=additional_val_name, file_writer=file_writer))

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=base_logdir)
    callbacks.append(tb_callback)
    log_callback = LogCallback(epochs, log)
    callbacks.append(log_callback)

    if checkpoint is not None:
        os.makedirs(os.path.join(base_logdir, checkpoint), exist_ok=True)
        # base_checkpoints = BestNModelCheckpoint(filepath=os.path.join(base_logdir, checkpoint, 'model-{epoch:02d}-{val_binary_accuracy:.2f}.h5'),
        #                                         max_to_keep=2,
        #                                         monitor='val_binary_accuracy',
        #                                         mode='max',
        #                                         save_weights_only=True,
        #                                         save_best_only=True,
        #                                         save_freq='epoch',
        #                                         verbose=1,)
        base_checkpoints = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint,
                                                              monitor='val_binary_accuracy',
                                                              mode='max',
                                                              save_weights_only=False,
                                                              save_best_only=False,
                                                              save_freq='epoch',
                                                              verbose=1,)
        callbacks.append(base_checkpoints)

    if backup is not None:
        os.makedirs(os.path.join(base_logdir, backup), exist_ok=True)
        backup_callback = tf.keras.callbacks.BackupAndRestore(backup_dir=os.path.join(
            base_logdir, backup), delete_checkpoint=False, save_freq=backup_freq if backup_freq is not None else "epoch")
        callbacks.append(backup_callback)

    return callbacks
