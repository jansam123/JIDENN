import tensorflow as tf
import numpy as np
import pandas as pd
import os
import logging
import hydra
import copy
from functools import partial
from hydra.core.config_store import ConfigStore
#
from jidenn.config import config
import jidenn.data.data_info as data_info
from jidenn.model_builders.ModelBuilder import ModelBuilder
from jidenn.data.get_dataset import get_preprocessed_dataset
from jidenn.model_builders.callbacks import get_callbacks
from jidenn.evaluation.plotter import plot_train_history
from jidenn.model_builders.normalization_initialization import get_normalization
from jidenn.data.TrainInput import input_classes_lookup
from jidenn.model_builders.multi_gpu_strategies import choose_strategy
from jidenn.data.augmentations import construct_augmentation

cs = ConfigStore.instance()
cs.store(name="args", node=config.JIDENNConfig)


@hydra.main(version_base="1.2", config_path="jidenn/yaml_config", config_name="config")
def main(args: config.JIDENNConfig) -> None:
    log = logging.getLogger(__name__)

    # GPU logging
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) == 0:
        log.warning("No GPU found, using CPU")
    for i, gpu in enumerate(gpus):
        gpu_info = tf.config.experimental.get_device_details(gpu)
        log.info(
            f"GPU {i}: {gpu_info['device_name']} with compute capability {gpu_info['compute_capability'][0]}.{gpu_info['compute_capability'][1]}")

    # CUDA logging
    system_config = tf.sysconfig.get_build_info()
    log.info(f"System Config: {system_config}")

    gpu_strategy = partial(choose_strategy, num_gpus=len(gpus))

    # debug mode for tensorflow
    if args.general.debug:
        log.info("Debug mode enabled")
        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()

    # fixing seed for reproducibility
    if args.general.seed is not None:
        log.info(f"Setting seed to {args.general.seed}")
        np.random.seed(args.general.seed)
        tf.random.set_seed(args.general.seed)

    # managing threads
    if args.general.threads is not None:
        tf.config.threading.set_inter_op_parallelism_threads(
            args.general.threads)
        tf.config.threading.set_intra_op_parallelism_threads(
            args.general.threads)

    # set decay steps based on the total number of iterations
    if args.optimizer.decay_steps is None and args.dataset.take is not None:
        args.optimizer.decay_steps = int(args.dataset.epochs * args.dataset.take /
                                         args.dataset.batch_size) - args.optimizer.warmup_steps
        log.info(f"Setting decay steps to {args.optimizer.decay_steps}")
    elif args.optimizer.decay_steps is None and args.dataset.take is None:
        raise ValueError("Cannot set decay steps if dataset.take is None")
    else:
        log.info(f"Decay steps set to {args.optimizer.decay_steps}")

    # pick input variables according to model
    # if you want to choose your own input, implement a subclass of `TrainInput` in  `jidenn.data.TrainInput` and put it into the dict in the function `input_classes_lookup`
    train_input_class = input_classes_lookup(
        getattr(args.models, args.general.model).train_input)
    train_input_class = train_input_class()
    model_input_creator = tf.function(func=train_input_class)
    input_shape = train_input_class.input_shape

    augmentation_function = construct_augmentation(
        args.augmentations) if args.augmentations is not None else None
    log.info(f"Using augmentations: {args.augmentations.order}") if args.augmentations is not None else log.warning(
        "No augmentations specified")

    train_path = [os.path.join(path, 'train') for path in list(args.data.path)] if not isinstance(
        args.data.path, str) else os.path.join(args.data.path, 'train')
    dev_path = [os.path.join(path, 'dev') for path in list(args.data.path)] if not isinstance(
        args.data.path, str) else os.path.join(args.data.path, 'dev')
    args_data_train = copy.deepcopy(args.data)
    args_data_train.path = train_path
    args_data_dev = copy.deepcopy(args.data)
    args_data_dev.path = dev_path
    train = get_preprocessed_dataset(
        args_data_train, input_creator=model_input_creator, augmentation=augmentation_function)
    dev = get_preprocessed_dataset(
        args_data_dev, input_creator=model_input_creator)

    if args.test_data is not None:
        test = get_preprocessed_dataset(
            args.test_data, input_creator=model_input_creator)
    else:
        test = None

    try:
        restoring_from_backup = len(os.listdir(os.path.join(
            args.general.logdir, args.general.backup))) > 0
    except FileNotFoundError:
        restoring_from_backup = False

    # draw input data distribution
    if args.preprocess.draw_distribution is not None and args.preprocess.draw_distribution > 0 and not restoring_from_backup:
        log.info(
            f"Drawing data distribution with {args.preprocess.draw_distribution} samples")
        dir = os.path.join(args.general.logdir, 'dist')
        os.makedirs(dir, exist_ok=True)
        named_labels = {0: args.data.labels[0], 1: args.data.labels[1]}
        train.take(args.preprocess.draw_distribution).plot_data_distributions(hue_variable='label',
                                                                              folder=dir, named_labels=named_labels)

    # get proper dataset size based on the config
    if args.dataset.take is not None:
        train_size = 1 - args.dataset.dev_size
        train_size = int(train_size * args.dataset.take)
        dev_size = int(args.dataset.dev_size * args.dataset.take)
    else:
        train_size, dev_size = None, None

    # get fully prepared (batched, shuffled, prefetched) dataset
    train = train.get_prepared_dataset(batch_size=args.dataset.batch_size,
                                       shuffle_buffer_size=args.dataset.shuffle_buffer,
                                       take=train_size,
                                       ragged=False if args.general.model == 'particlenet' else True,
                                       assert_length=True)
    dev = dev.get_prepared_dataset(batch_size=args.dataset.batch_size,
                                   ragged=False if args.general.model == 'particlenet' else True,
                                   take=dev_size)

    test = test.get_prepared_dataset(batch_size=args.dataset.batch_size,
                                     ragged=False if args.general.model == 'particlenet' else True,
                                     take=args.dataset.test_take) if test is not None else None

    # this is only to get rid of some warnings
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train = train.with_options(options)
    dev = dev.with_options(options)
    test = test.with_options(options) if test is not None else None

    # build model, gpu_strategy is based on the number of gpus available
    @gpu_strategy
    def build_model() -> tf.keras.Model:

        # create and adapt a nomralization layer
        # this helps with faster convergence of the models
        if args.preprocess.normalization_size is not None and args.preprocess.normalization_size > 0 and args.general.model != 'bdt':
            adapt = True if args.general.load_checkpoint_path is None else False
            adapt = adapt and not restoring_from_backup
            normalizer = get_normalization(dataset=train,
                                           adapt=True,
                                           input_shape=input_shape,
                                           normalization_steps=args.preprocess.normalization_size,
                                           log=log)
        else:
            normalizer = None
            log.warning("Normalization disabled.")

        # create a ModelBuilder class construct the model specified
        model_builder = ModelBuilder(model_name=args.general.model,
                                     args_model=args.models,
                                     input_size=input_shape,
                                     num_labels=len(args.data.labels),
                                     args_optimizer=args.optimizer,
                                     preprocess=normalizer,
                                     )
        # obtain the compiled model
        model = model_builder.compiled_model

        if args.general.model != 'bdt':
            model.summary(print_fn=log.info, line_length=120,
                          show_trainable=True)
        else:
            log.warning("No model summary for BDT")
        return model

    # build the model
    model = build_model()

    if args.general.load_checkpoint_path is not None:
        model.load_weights(args.general.load_checkpoint_path)

    # get callbacks based on the input

    if args.general.model == 'bdt' and args.dataset.epochs > 1:
        log.warning("BDT does not support multiple epochs. Setting epochs to 1")
        args.dataset.epochs = 1

    if args.dataset.cache == 'mem':
        train = train.cache()
        dev = dev.cache()
        test = test.cache() if test is not None else None
    elif args.dataset.cache == 'disk':
        os.makedirs(f'{args.general.logdir}/cache/train', exist_ok=True)
        os.makedirs(f'{args.general.logdir}/cache/dev', exist_ok=True)
        train = train.cache(f'{args.general.logdir}/cache/train')
        dev = dev.cache(f'{args.general.logdir}/cache/dev')
        test = test.cache(
            f'{args.general.logdir}/cache/test') if test is not None else None

    callbacks = get_callbacks(base_logdir=args.general.logdir,
                              epochs=args.dataset.epochs,
                              log=log,
                              backup=args.general.backup,
                              backup_freq=args.general.backup_freq,
                              checkpoint=os.path.join(
                                  args.general.logdir, 'checkpoint'),
                              additional_val_dataset=test,
                              additional_val_name='test')
    # running training
    train = train.apply(tf.data.experimental.assert_cardinality(
        args.dataset.steps_per_epoch)) if args.dataset.steps_per_epoch is not None else train
    history = model.fit(train,
                        epochs=args.dataset.epochs,
                        callbacks=callbacks,
                        validation_data=dev,
                        # steps_per_epoch=args.dataset.steps_per_epoch,
                        verbose=2 if args.general.model == 'bdt' else 1)

    model.load_weights(os.path.join(args.general.logdir, 'checkpoint'))

    # saving the model
    model_dir = os.path.join(args.general.logdir, 'model')
    log.info(f"Saving model to {model_dir}")
    model.save(model_dir, save_format='tf')

    # save the training history and plot it
    if args.general.model != 'bdt':
        log.info(f"Saving history")
        history_dir = os.path.join(args.general.logdir, 'history')
        os.makedirs(history_dir, exist_ok=True)
        for metric in model.metrics_names:
            metric_dict = {
                f'{metric}': history.history[metric], f'validation {metric}': history.history[f'val_{metric}']}
            if f'test_{metric}' in history.history.keys():
                metric_dict[f'test {metric}'] = history.history[f'test_{metric}']
            plot_train_history(metric_dict, history_dir,
                               metric, args.dataset.epochs)

        # for metric in [m for m in history.history.keys() if 'val' not in m]:
        #     plot_train_history(
        #         {f'{metric}': history.history[metric], f'validation {metric}': history.history[f'val_{metric}']}, history_dir, metric, args.dataset.epochs)

    # run simple evaluation
    log.info("Evaluating on dev set:")
    log.info(model.evaluate(dev, return_dict=True))
    if args.test_data is not None:
        log.info("Evaluating on test set:")
        log.info(model.evaluate(test, return_dict=True))

    # if model is bdt, plot feature importances
    if args.general.model == 'bdt':
        model.summary(print_fn=log.info)
        variable_importance_metric = "SUM_SCORE"
        variable_importances = model.make_inspector().variable_importances()[
            variable_importance_metric]
        variable_importances = pd.DataFrame({'variable': [str(vi[0].name) for vi in variable_importances],
                                             'score': [vi[1] for vi in variable_importances]})
        variables_names = ['pt_jet', 'eta_jet',
                           'N_PFO', 'W_PFO_jet', 'C1_PFO_jet']
        variable_importances['variable'] = variable_importances['variable'].apply(
            lambda x: variables_names[int(x.split('.')[-1])])
        data_info.plot_feature_importance(variable_importances, os.path.join(
            args.general.logdir, f'feature_bdt_score.png'))
    log.info("Done!")


if __name__ == "__main__":
    main()
