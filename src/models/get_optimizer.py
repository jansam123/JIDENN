import tensorflow as tf
import tensorflow_addons as tfa

from src.config import config_subclasses as cfg
from .optimizers import LinearWarmup

def get_optimizer(args_optimizer: cfg.Optimizer) -> tf.keras.optimizers.Optimizer:
    
    optimizer = 'Adam' if args_optimizer.name is None else args_optimizer.name
    learning_rate = 0.001 if args_optimizer.learning_rate is None else args_optimizer.learning_rate
    decay_steps = None if args_optimizer.decay_steps is None else args_optimizer.decay_steps
    warmup_steps = None if args_optimizer.warmup_steps is None else args_optimizer.warmup_steps
    beta_1 = 0.9 if args_optimizer.beta_1 is None else args_optimizer.beta_1
    beta_2 = 0.999 if args_optimizer.beta_2 is None else args_optimizer.beta_2
    epsilon = 1e-6 if args_optimizer.epsilon is None else args_optimizer.epsilon
    clipnorm = None if args_optimizer.clipnorm is None else args_optimizer.clipnorm
    weight_decay = 0.0 if args_optimizer.weight_decay is None else args_optimizer.weight_decay
    l_r = tf.keras.optimizers.schedules.CosineDecay(
        learning_rate, decay_steps) if decay_steps is not None else learning_rate

    if warmup_steps is not None:
        l_r = LinearWarmup(warmup_steps, l_r)

    if optimizer == 'LAMB':
        return tfa.optimizers.LAMB(learning_rate=l_r,
                                   weight_decay=weight_decay,
                                   beta_1=beta_1,
                                   beta_2=beta_2,
                                   epsilon=epsilon,
                                   clipnorm=clipnorm)
    elif optimizer == 'Adam':
        if weight_decay > 0:
            return tfa.optimizers.AdamW(learning_rate=l_r,
                                        weight_decay=weight_decay,
                                        beta_1=beta_1,
                                        beta_2=beta_2,
                                        epsilon=epsilon,
                                        clipnorm=clipnorm)
        else:
            return tf.optimizers.Adam(learning_rate=l_r,
                                    beta_1=beta_1,
                                    beta_2=beta_2,
                                    epsilon=epsilon,
                                    clipnorm=clipnorm)
    else:
        raise NotImplementedError(f'Optimizer {optimizer} not supported.')
