import tensorflow as tf
from datetime import datetime
from src.config import config_subclasses as cfg
from logging import Logger

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
        log_str +=  " - ".join([f"{k}: {v:.4}" for k,v in logs.items()])
        self._log.info(log_str)

class ValidationCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_dataset: tf.data.Dataset, log: Logger):
        self._validation_dataset = validation_dataset
        self._log = log
        
    def on_batch_end(self, batch, logs=None):
        if (batch + 1) % 1000 != 0:
            return
        metrics = self.model.evaluate(self._validation_dataset, verbose=0)
        log_str = f"Validation: "
        log_str +=  " - ".join([f"{k}: {v:.4}" for k,v in zip(self.model.metrics_names, metrics)])
        self._log.info(log_str)
        for k,v in zip(self.model.metrics_names, metrics):
            logs[f'val_{k}'] = v
            
def get_callbacks(args: cfg.Params, log: Logger, val_dataset: tf.data.Dataset) -> list[tf.keras.callbacks.Callback]:
    callbacks = []
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=args.logdir, histogram_freq=1, embeddings_freq=1, profile_batch=(10,15))
    callbacks += [tb_callback]
            
    callbacks += [LogCallback(args.epochs, log)]#, ValidationCallback(val_dataset, log)]
    
    return callbacks