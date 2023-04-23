Module jidenn.model_builders.LearningRateSchedulers
===================================================
Module for custom Learning Rate Schedulers.

Classes
-------

`ConstantWarmup(warmup_steps: int, following_schedule: keras.optimizers.schedules.learning_rate_schedule.LearningRateSchedule)`
:   Constant warmup schedule.
    Keeps learning rate at constant value (first step of following schedule)for warmup_steps, then follows following schedule.
    
    Args:
        warmup_steps (int): Number of warmup steps.
        following_schedule (tf.optimizers.schedules.LearningRateSchedule): Following schedule.

    ### Ancestors (in MRO)

    * keras.optimizers.schedules.learning_rate_schedule.LearningRateSchedule

    ### Methods

    `get_config(self)`
    :

`LinearWarmup(warmup_steps: int, following_schedule: keras.optimizers.schedules.learning_rate_schedule.LearningRateSchedule)`
:   Linear warmup schedule.
    Linearly increases learning rate from 0 to following schedule's first value over warmup_steps.
    
    Args:
        warmup_steps (int): Number of warmup steps.
        following_schedule (tf.optimizers.schedules.LearningRateSchedule): Following schedule.

    ### Ancestors (in MRO)

    * keras.optimizers.schedules.learning_rate_schedule.LearningRateSchedule

    ### Methods

    `get_config(self)`
    :