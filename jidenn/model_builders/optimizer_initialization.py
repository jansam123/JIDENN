"""
Module for initializing the optimizer from the config file.
The corresponding config dataclass is defined in `jidenn.config.config.Optimizer`.
"""
import tensorflow as tf
import keras
# import tensorflow_addons as tfa

from jidenn.config import config
from .LearningRateSchedulers import LinearWarmup

def get_optimizer(args_optimizer: config.Optimizer) -> keras.optimizers.Optimizer:
    """Initializes the optimizer from the config file.
    Possible optimizers are `tf.optimizers.Adam` and `tf.optimizers.LAMB`.
    If the `weight_decay` parameter is set, the `tf.optimizers.AdamW` optimizer is used.

    Args:
        args_optimizer (jidenn.config.config.Optimizer): config dataclass for the optimizer

    Raises:
        NotImplementedError: Raised if the optimizer is not supported.

    Returns:
        keras.optimizers.Optimizer: Optimizer object with set parameters.
    """

    optimizer = 'Adam' if args_optimizer.name is None else args_optimizer.name
    learning_rate = 0.001 if args_optimizer.learning_rate is None else args_optimizer.learning_rate
    decay_steps = None if args_optimizer.decay_steps is None else args_optimizer.decay_steps
    warmup_steps = None if args_optimizer.warmup_steps is None else args_optimizer.warmup_steps
    beta_1 = 0.9 if args_optimizer.beta_1 is None else args_optimizer.beta_1
    beta_2 = 0.999 if args_optimizer.beta_2 is None else args_optimizer.beta_2
    epsilon = 1e-6 if args_optimizer.epsilon is None else args_optimizer.epsilon
    clipnorm = None if args_optimizer.clipnorm is None or args_optimizer.clipnorm == 0.0 else args_optimizer.clipnorm
    weight_decay = 0.0 if args_optimizer.weight_decay is None else args_optimizer.weight_decay
    min_lr = 0.0 if args_optimizer.min_learning_rate is None else args_optimizer.min_learning_rate
    clipvalue = None if args_optimizer.clipvalue is None or args_optimizer.clipvalue == 0.0 else args_optimizer.clipvalue
    
    l_r = keras.optimizers.schedules.CosineDecay(
        learning_rate, decay_steps, alpha=min_lr) if decay_steps is not None else learning_rate

    if warmup_steps is not None:
        l_r = LinearWarmup(warmup_steps, l_r)
    

    if optimizer == 'Lamb':
        print('Clipvalue: ',clipvalue)
        return keras.optimizers.Lamb(learning_rate=l_r,
                                   weight_decay=weight_decay,
                                   beta_1=beta_1,
                                   beta_2=beta_2,
                                   epsilon=epsilon,
                                   clipnorm=clipnorm,
                                   clipvalue=clipvalue,
                                   use_ema=True,
                                   )
    elif optimizer == 'Lion':
        return keras.optimizers.Lion(learning_rate=l_r,
                    weight_decay=weight_decay,
                    beta_1=beta_1,
                    beta_2=beta_2,
                    clipvalue=clipvalue,
                    clipnorm=clipnorm)

    elif optimizer == 'Adam':
        return keras.optimizers.Adam(learning_rate=l_r,
                                    beta_1=beta_1,
                                    beta_2=beta_2,
                                    weight_decay=weight_decay,
                                    epsilon=epsilon,
                                    clipvalue=clipvalue,
                                    clipnorm=clipnorm)
    elif optimizer == 'AdamW':
        if weight_decay == 0.0:
            raise ValueError('AdamW requires weight_decay > 0.0.')
        return keras.optimizers.AdamW(learning_rate=l_r,
                     beta_1=beta_1,
                     beta_2=beta_2,
                     weight_decay=weight_decay,
                     epsilon=epsilon,
                     clipvalue=clipvalue,
                     clipnorm=clipnorm)
    elif optimizer == 'NAdam':
        return keras.optimizers.Nadam(learning_rate=l_r,
                     beta_1=beta_1,
                     beta_2=beta_2,
                     epsilon=epsilon,
                     clipnorm=clipnorm,
                     clipvalue=clipvalue,
                     weight_decay=weight_decay)
    else:
        raise NotImplementedError(f'Optimizer {optimizer} not supported.')
