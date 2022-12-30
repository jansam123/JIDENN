import tensorflow as tf
import tensorflow_addons as tfa


class LinearWarmup(tf.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps, following_schedule):
        self._warmup_steps = warmup_steps
        self._warmup = tf.optimizers.schedules.PolynomialDecay(0., warmup_steps, following_schedule(0))
        self._following = following_schedule

    def get_config(self):
        config = {
            'warmup_steps': self._warmup_steps,
            'following_schedule': self._following
        }
        return config

    def __call__(self, step):
        return tf.cond(step < self._warmup_steps,
                       lambda: self._warmup(step),
                       lambda: self._following(step - self._warmup_steps))


class ConstantWarmup(tf.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps, following_schedule):
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
        return tf.cond(step < self._warmup_steps,
                       lambda: self._warmup,
                       lambda: self._following(step - self._warmup_steps))
