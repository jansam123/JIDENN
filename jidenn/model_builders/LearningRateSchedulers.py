"""
Module for custom Learning Rate Schedulers.
"""
import tensorflow as tf


class LinearWarmup(tf.optimizers.schedules.LearningRateSchedule):
    """Linear warmup schedule.
    Linearly increases learning rate from 0 to following schedule's first value over warmup_steps.

    Args:
        warmup_steps (int): Number of warmup steps.
        following_schedule (tf.optimizers.schedules.LearningRateSchedule): Following schedule.
    """

    def __init__(self, warmup_steps: int, following_schedule: tf.optimizers.schedules.LearningRateSchedule):
        self._warmup_steps = warmup_steps
        self._warmup = tf.optimizers.schedules.PolynomialDecay(0., warmup_steps, following_schedule(0))
        self._following = following_schedule

    def get_config(self):
        config = {
            'warmup_steps': self._warmup_steps,
            'following_schedule': self._following
        }
        return config

    def __call__(self, step: int):
        """Executes learning rate schedule.

        Args:
            step (int): Current step.

        Returns:
            Learning rate.
        """
        return tf.cond(step < self._warmup_steps,
                       lambda: self._warmup(step),
                       lambda: self._following(step - self._warmup_steps))


class ConstantWarmup(tf.optimizers.schedules.LearningRateSchedule):
    """Constant warmup schedule.
    Keeps learning rate at constant value (first step of following schedule)for warmup_steps, then follows following schedule.

    Args:
        warmup_steps (int): Number of warmup steps.
        following_schedule (tf.optimizers.schedules.LearningRateSchedule): Following schedule.
    """

    def __init__(self, warmup_steps: int, following_schedule: tf.optimizers.schedules.LearningRateSchedule):
        self._warmup_steps = warmup_steps
        self._warmup = following_schedule(0)
        self._following = following_schedule

    def get_config(self):
        config = {
            'warmup_steps': self._warmup_steps,
            'following_schedule': self._following
        }
        return config

    def __call__(self, step):
        """Executes learning rate schedule.
        Args:
            step (int): Current step.

        Returns:
            Learning rate.
        """
        return tf.cond(step < self._warmup_steps,
                       lambda: self._warmup,
                       lambda: self._following(step - self._warmup_steps))
