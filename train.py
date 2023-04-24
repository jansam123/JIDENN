import tensorflow as tf
import numpy as np
import pandas as pd
import os
import logging
import hydra
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

cs = ConfigStore.instance()
cs.store(name="args", node=config.JIDENNConfig)


@hydra.main(version_base="1.2", config_path="jidenn/config", config_name="config")
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
    
    #CUDA logging
    cuda_version = tf.sysconfig.get_build_info()["cuda_version"]
    log.info(f"CUDA version: {cuda_version}")


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
        tf.config.threading.set_inter_op_parallelism_threads(args.general.threads)
        tf.config.threading.set_intra_op_parallelism_threads(args.general.threads)

    # set decay steps
    if args.optimizer.decay_steps is None and args.dataset.take is not None:
        args.optimizer.decay_steps = int(args.dataset.epochs * args.dataset.take /
                                         args.dataset.batch_size) - args.optimizer.warmup_steps
        log.info(f"Setting decay steps to {args.optimizer.decay_steps}")
    elif args.optimizer.decay_steps is None and args.dataset.take is None:
        raise ValueError("Cannot set decay steps if dataset.take is None")
    else:
        log.info(f"Decay steps set to {args.optimizer.decay_steps}")

    files_per_JZ_slice = []
    for name in ["train", "test", "dev"]:
        file = [f'{args.data.path}/{jz_slice}/{name}' for jz_slice in args.data.subfolders] if args.data.subfolders is not None else [f'{args.data.path}/{name}']
        files_per_JZ_slice.append(file)

    train, test, dev = [get_preprocessed_dataset(file, args.data) for file in files_per_JZ_slice]

    # pick input variables according to model

    train_input_class = input_classes_lookup(getattr(args.models, args.general.model).train_input)
    train_input_class = train_input_class(per_jet_variables=args.data.variables.per_jet,
                                          per_event_variables=args.data.variables.per_event,
                                          per_jet_tuple_variables=args.data.variables.per_jet_tuple)
    model_input = tf.function(func=train_input_class)
    input_size = train_input_class.input_shape

    train = train.create_train_input(model_input)
    dev = dev.create_train_input(model_input)
    test = test.create_train_input(model_input)

    # draw input data distribution
    if args.preprocess.draw_distribution is not None and args.preprocess.draw_distribution > 0:
        log.info(f"Drawing data distribution with {args.preprocess.draw_distribution} samples")
        dir = os.path.join(args.general.logdir, 'dist')
        os.makedirs(dir, exist_ok=True)

        dist_dataset = train.apply(lambda x: x.take(args.preprocess.draw_distribution))
        df = dist_dataset.to_pandas()
        df['named_label'] = df['label'].replace({0: args.data.labels[0], 1: args.data.labels[1]})
        data_info.generate_data_distributions(df=df, folder=dir)

    # get proper dataset size
    if args.dataset.take is not None:
        train_size = 1 - args.dataset.dev_size + args.dataset.test_size
        train_size = int(train_size * args.dataset.take)
        dev_size = int(args.dataset.dev_size * args.dataset.take)
        test_size = int(args.dataset.test_size * args.dataset.take)
    else:
        train_size, dev_size, test_size = None, None, None

    # get fully prepared (batched, shuffled, prefetched) dataset
    train = train.get_prepared_dataset(batch_size=args.dataset.batch_size,
                                       shuffle_buffer_size=args.dataset.shuffle_buffer,
                                       take=train_size,
                                       assert_length=True)
    dev = dev.get_prepared_dataset(batch_size=args.dataset.batch_size,
                                   take=dev_size)
    test = test.get_prepared_dataset(batch_size=args.dataset.batch_size,
                                     take=test_size)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train = train.with_options(options)
    dev = dev.with_options(options)
    test = test.with_options(options)

    # build model
    @gpu_strategy
    def build_model() -> tf.keras.Model:
        if args.preprocess.normalization_size is not None and args.preprocess.normalization_size > 0 and args.general.model != 'bdt':
            adapt = True if args.general.load_checkpoint_path is None else False
            normalizer = get_normalization(dataset=train,
                                           adapt=adapt,
                                           ragged=isinstance(input_size, tuple),
                                           normalization_steps=args.preprocess.normalization_size,
                                           interaction=isinstance(input_size, tuple) and isinstance(
                                               input_size[0], tuple),
                                           log=log)
        else:
            normalizer = None
            log.warning("Normalization disabled.")

        model_builder = ModelBuilder(model_name=args.general.model,
                                     args_model=args.models,
                                     input_size=input_size,
                                     num_labels=len(args.data.labels),
                                     args_optimizer=args.optimizer,
                                     preprocess=normalizer,
                                     )

        model = model_builder.compiled_model

        if args.general.model != 'bdt':
            model.summary(print_fn=log.info, line_length=120, show_trainable=True)
        else:
            log.warning("No model summary for BDT")
        return model

    model = build_model()

    # TODO1: rename load_checkpoint_path to load_weights_path for clarity
    # TODO2: add option to load model instead of weights

    if args.general.load_checkpoint_path is not None:
        model.load_weights(args.general.load_checkpoint_path)

    # callbacks
    callbacks = get_callbacks(args.general.logdir, args.dataset.epochs, log,
                              args.general.checkpoint, args.general.backup)

    # running training
    history = model.fit(train,
                        epochs=args.dataset.epochs,
                        callbacks=callbacks,
                        validation_data=dev,
                        verbose=2 if args.general.model == 'bdt' else 1)

    # saving model
    model_dir = os.path.join(args.general.logdir, 'model')
    log.info(f"Saving model to {model_dir}")
    model.save(model_dir, save_format='tf')

    if args.general.model != 'bdt':
        log.info(f"Saving history")
        history_dir = os.path.join(args.general.logdir, 'history')
        os.makedirs(history_dir, exist_ok=True)
        for metric in [m for m in history.history.keys() if 'val' not in m]:
            plot_train_history(
                {f'{metric}': history.history[metric], f'validation {metric}': history.history[f'val_{metric}']}, history_dir, metric, args.dataset.epochs)

    if test is None:
        log.error("No test dataset, skipping evaluation.")
        log.info("Done!")
        return

    log.info(model.evaluate(test, return_dict=True))

    if args.general.model == 'bdt':
        model.summary(print_fn=log.info)
        variable_importance_metric = "SUM_SCORE"
        variable_importances = model.make_inspector().variable_importances()[variable_importance_metric]
        variable_importances = pd.DataFrame({'variable': [str(vi[0].name) for vi in variable_importances],
                                             'score': [vi[1] for vi in variable_importances]})
        variables_names = ['pt_jet', 'eta_jet', 'N_PFO', 'W_PFO_jet', 'C1_PFO_jet']
        variable_importances['variable'] = variable_importances['variable'].apply(
            lambda x: variables_names[int(x.split('.')[-1])])
        data_info.plot_feature_importance(variable_importances, os.path.join(
            args.general.logdir, f'feature_bdt_score.png'))
    log.info("Done!")


if __name__ == "__main__":
    main()
