import tensorflow as tf
from datetime import datetime
from src.config import config_subclasses as cfg
from logging import Logger



def get_callbacks(args: cfg.Params, log: Logger) -> list[tf.keras.callbacks.Callback]:
    callbacks = []
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=args.logdir, histogram_freq=1, embeddings_freq=1)
    callbacks += [tb_callback]
    
    class LogCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            self.__epoch_start = None
        
        def on_epoch_begin(self, epoch, logs=None):
            self.__epoch_start = datetime.utcnow()
            
        def on_epoch_end(self, epoch, logs):
            eta = datetime.utcnow() - self.__epoch_start if self.__epoch_start is not None else '-:--:--'
            # convert from timedelta to datetime
            eta = datetime.strptime(str(eta), "%H:%M:%S.%f")
            
            log_str = f"Epoch {epoch+1}/{args.epochs}: "
            log_str += f"ETA: {eta:%T} - "
            log_str +=  " - ".join([f"{k}: {v:.4}" for k,v in logs.items()])
            log.info(log_str)
            
    callbacks += [LogCallback()]
    
    return callbacks