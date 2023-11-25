"""
Module for initializing the optimizer from the config file.
The corresponding config dataclass is defined in `jidenn.config.config.Optimizer`.
"""
import tensorflow as tf
# import tensorflow_addons as tfa

from jidenn.config import config
from .LearningRateSchedulers import LinearWarmup


class Lion(tf.keras.optimizers.Optimizer):
    """Original source: https://github.com/keras-team/keras/blob/master/keras/optimizers/lion.py
    Optimizer that implements the Lion algorithm.

    The Lion optimizer is a stochastic-gradient-descent method that uses the
    sign operator to control the magnitude of the update, unlike other adaptive
    optimizers such as Adam that rely on second-order moments. This make
    Lion more memory-efficient as it only keeps track of the momentum. According
    to the authors (see reference), its performance gain over Adam grows with
    the batch size. Because the update of Lion is produced through the sign
    operation, resulting in a larger norm, a suitable learning rate for Lion is
    typically 3-10x smaller than that for AdamW. The weight decay for Lion
    should be in turn 3-10x larger than that for AdamW to maintain a
    similar strength (lr * wd).

    Args:
      learning_rate: A `tf.Tensor`, floating point value, a schedule that is a
        `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
        that takes no arguments and returns the actual value to use. The
        learning rate. Defaults to 0.0001.
      beta_1: A float value or a constant float tensor, or a callable
        that takes no arguments and returns the actual value to use. The rate
        to combine the current gradient and the 1st moment estimate.
      beta_2: A float value or a constant float tensor, or a callable
        that takes no arguments and returns the actual value to use. The
        exponential decay rate for the 1st moment estimate.


    References:
      - [Chen et al., 2023](http://arxiv.org/abs/2302.06675)
      - [Authors' implementation](
          http://github.com/google/automl/tree/master/lion)

    """

    def __init__(
        self,
        learning_rate=0.0001,
        beta_1=0.9,
        beta_2=0.99,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        jit_compile=True,
        name="Lion",
        **kwargs,
    ):
        super().__init__(
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            jit_compile=jit_compile,
            **kwargs,
        )
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        if beta_1 <= 0 or beta_1 > 1:
            raise ValueError(
                f"`beta_1`={beta_1} must be between ]0, 1]. Otherwise, "
                "the optimizer degenerates to SignSGD."
            )

    def build(self, var_list):
        """Initialize optimizer variables.

        Lion optimizer has one variable `momentums`.

        Args:
          var_list: list of model variables to build Lion variables on.
        """
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self.momentums = []
        for var in var_list:
            self.momentums.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="m"
                )
            )
        self._built = True

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        lr = tf.cast(self.learning_rate, variable.dtype)
        beta_1 = tf.cast(self.beta_1, variable.dtype)
        beta_2 = tf.cast(self.beta_2, variable.dtype)
        var_key = self._var_key(variable)
        m = self.momentums[self._index_dict[var_key]]

        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients (use m as a buffer)
            m.assign(m * beta_1)
            m.scatter_add(
                tf.IndexedSlices(
                    gradient.values * (1.0 - beta_1), gradient.indices
                )
            )
            variable.assign_sub(lr * tf.math.sign(m))

            m.assign(m * beta_2 / beta_1)
            m.scatter_add(
                tf.IndexedSlices(
                    gradient.values * (1.0 - beta_2 / beta_1), gradient.indices
                )
            )
        else:
            # Dense gradients
            variable.assign_sub(
                lr * tf.math.sign(m * beta_1 + gradient * (1.0 - beta_1))
            )
            m.assign(m * beta_2 + gradient * (1.0 - beta_2))

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    self._learning_rate
                ),
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
            }
        )
        return config


def get_optimizer(args_optimizer: config.Optimizer) -> tf.keras.optimizers.Optimizer:
    """Initializes the optimizer from the config file.
    Possible optimizers are `tf.optimizers.Adam` and `tf.optimizers.LAMB`.
    If the `weight_decay` parameter is set, the `tf.optimizers.AdamW` optimizer is used.

    Args:
        args_optimizer (jidenn.config.config.Optimizer): config dataclass for the optimizer

    Raises:
        NotImplementedError: Raised if the optimizer is not supported.

    Returns:
        tf.keras.optimizers.Optimizer: Optimizer object with set parameters.
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
    
    l_r = tf.keras.optimizers.schedules.CosineDecay(
        learning_rate, decay_steps, alpha=min_lr) if decay_steps is not None else learning_rate

    if warmup_steps is not None:
        l_r = LinearWarmup(warmup_steps, l_r)

    if optimizer == 'LAMB':
        raise NotImplementedError('LAMB optimizer not supported.')
        # return tfa.optimizers.LAMB(learning_rate=l_r,
        #                            weight_decay=weight_decay,
        #                            beta_1=beta_1,
        #                            beta_2=beta_2,
        #                            epsilon=epsilon,
        #                            clipnorm=clipnorm)
    elif optimizer == 'Lion':
        return Lion(learning_rate=l_r,
                    weight_decay=weight_decay,
                    beta_1=beta_1,
                    beta_2=beta_2,
                    clipnorm=clipnorm)

    elif optimizer == 'Adam':
        return tf.optimizers.Adam(learning_rate=l_r,
                                    beta_1=beta_1,
                                    beta_2=beta_2,
                                    weight_decay=weight_decay,
                                    epsilon=epsilon,
                                    clipnorm=clipnorm)
    else:
        raise NotImplementedError(f'Optimizer {optimizer} not supported.')
