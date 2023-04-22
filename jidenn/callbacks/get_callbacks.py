import tensorflow as tf
import os
from datetime import datetime
from logging import Logger
from typing import List, Optional

from jidenn.config import config_subclasses as cfg
from .BestNModelCheckpoint import BestNModelCheckpoint


class LogCallback(tf.keras.callbacks.Callback):
    def __init__(self, epochs: int, log: Logger):
        self.__epoch_start = None
        self._log = log
        self._epochs = epochs

    def on_epoch_begin(self, epoch, logs=None):
        self.__epoch_start = datetime.utcnow()

    def on_epoch_end(self, epoch, logs):
        eta = datetime.utcnow() - self.__epoch_start if self.__epoch_start is not None else '-:--:--'
        # convert from timedelta to datetime
        eta = datetime.strptime(str(eta), "%H:%M:%S.%f")

        log_str = f"Epoch {epoch+1}/{self._epochs}: "
        log_str += f"ETA: {eta:%T} - "
        log_str += " - ".join([f"{k}: {v:.4}" for k, v in logs.items()])
        self._log.info(log_str)


def get_callbacks(args: cfg.Params,
                  log: Logger,
                  checkpoint: Optional[str] = 'checkpoints',
                  backup: Optional[str] = 'backup') -> List[tf.keras.callbacks.Callback]:

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=args.logdir)
    log_callback = LogCallback(args.epochs, log)
    callbacks = [tb_callback, log_callback]

    if checkpoint is not None:
        os.makedirs(os.path.join(args.logdir, checkpoint), exist_ok=True)
        base_checkpoints = BestNModelCheckpoint(filepath=os.path.join(args.logdir, checkpoint, 'model-{epoch:02d}-{val_binary_accuracy:.2f}.h5'),
                                                max_to_keep=2,
                                                monitor='val_binary_accuracy',
                                                mode='max',
                                                save_weights_only=True,
                                                save_best_only=True,
                                                save_freq='epoch',
                                                verbose=1,)
        callbacks.append(base_checkpoints)

    if backup is not None:
        os.makedirs(os.path.join(args.logdir, backup), exist_ok=True)
        backup_callback = tf.keras.callbacks.BackupAndRestore(backup_dir=os.path.join(args.logdir, backup))
        callbacks.append(backup_callback)

    return callbacks
