import tensorflow as tf
import numpy as np
import pandas as pd
import os
import logging
import hydra
from hydra.core.config_store import ConfigStore
#
from src.data.get_dataset import get_preprocessed_dataset
import src.data.data_info as data_info
from src.callbacks.get_callbacks import get_callbacks
from src.config import config
from src.evaluation.train_history import plot_train_history
from src.models.get_model import get_compiled_model
from src.models.get_normalization import get_normalization
from src.data.get_train_input import get_train_input, get_input_shape
# os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


cs = ConfigStore.instance()
cs.store(name="args", node=config.JIDENNConfig)


@hydra.main(version_base="1.2", config_path="src/config", config_name="config")
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

    # debug mode for tensorflow
    if args.params.debug:
        log.info("Debug mode enabled")
        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()

    # fixing seed for reproducibility
    if args.params.seed is not None:
        log.info(f"Setting seed to {args.params.seed}")
        np.random.seed(args.params.seed)
        tf.random.set_seed(args.params.seed)

    # managing threads
    if args.params.threads is not None:
        tf.config.threading.set_inter_op_parallelism_threads(args.params.threads)
        tf.config.threading.set_intra_op_parallelism_threads(args.params.threads)

    files_per_JZ_slice = []
    for name in ["train", "test", "dev"]:
        file = [f'{args.data.path}/{jz_slice}/{name}' for jz_slice in args.data.JZ_slices] if args.data.JZ_slices is not None else [f'{args.data.path}/{name}']
        files_per_JZ_slice.append(file)

    train, test, dev = [get_preprocessed_dataset(file, args.data) for file in files_per_JZ_slice]

    # pick input variables according to model
    if args.params.model == 'part':
        interaction = args.models.part.interaction
    elif args.params.model == 'depart':
        interaction = args.models.depart.interaction
    else:
        interaction = None
    num_total_variables = len(args.data.variables.perJet)
    num_total_variables += len(args.data.variables.perEvent) if args.data.variables.perEvent is not None else 0
    model_input = get_train_input(args.params.model, interaction)
    input_size = get_input_shape(args.params.model, num_total_variables, interaction)
    train = train.map_data(model_input)
    dev = dev.map_data(model_input)
    test = test.map_data(model_input)

    # draw input data distribution
    if args.preprocess.draw_distribution is not None and args.preprocess.draw_distribution > 0:
        log.info(f"Drawing data distribution with {args.preprocess.draw_distribution} samples")
        dir = os.path.join(args.params.logdir, 'dist')
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
    train = train.get_dataset(batch_size=args.dataset.batch_size,
                              shuffle_buffer_size=args.dataset.shuffle_buffer,
                              take=train_size,
                              assert_shape=True)
    dev = dev.get_dataset(batch_size=args.dataset.batch_size,
                          take=dev_size)
    test = test.get_dataset(batch_size=args.dataset.batch_size,
                            take=test_size)

    def build_model():
        if args.preprocess.normalize:
            normalizer = get_normalization(model_name=args.params.model,
                                           dataset=train,
                                           normalization_steps=args.preprocess.normalization_size,
                                           interaction=interaction,
                                           log=log)
        else:
            normalizer = None
            log.warning("Normalization disabled.")

        model = get_compiled_model(args.params.model,
                                   input_size=input_size,
                                   args_models=args.models,
                                   args_optimizer=args.optimizer,
                                   num_labels=args.data.num_labels,
                                   preprocess=normalizer)
        if args.params.model != 'bdt':
            model.summary(print_fn=log.info, line_length=120, show_trainable=True)
        else:
            log.warning("No model summary for BDT")
        return model

    # creating model
    if len(gpus) < 2:
        model = build_model()
    else:
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            model = build_model()

    # callbacks
    callbacks = get_callbacks(args.params, log, dev)

    # running training
    history = model.fit(train, epochs=args.params.epochs, callbacks=callbacks,
                        validation_data=dev, verbose=2 if args.params.model == 'bdt' else 1)

    # saving model
    model_dir = os.path.join(args.params.logdir, 'model')
    log.info(f"Saving model to {model_dir}")
    model.save(model_dir, save_format='tf')

    if args.params.model != 'bdt':
        log.info(f"Saving history")
        history_dir = os.path.join(args.params.logdir, 'history')
        os.makedirs(history_dir, exist_ok=True)
        for metric in [m for m in history.history.keys() if 'val' not in m]:
            plot_train_history(
                {f'{metric}': history.history[metric], f'validation {metric}': history.history[f'val_{metric}']}, history_dir, metric, args.params.epochs)

    if test is None:
        log.error("No test dataset, skipping evaluation.")
        log.info("Done!")
        return

    print(model.evaluate(test, return_dict=True))

    if args.params.model == 'bdt':
        model.summary(print_fn=log.info)
        variable_importance_metric = "SUM_SCORE"
        variable_importances = model.make_inspector().variable_importances()[variable_importance_metric]
        variable_importances = pd.DataFrame({'variable': [str(vi[0].name) for vi in variable_importances],
                                             'score': [vi[1] for vi in variable_importances]})
        variables_names = ['pt_jet', 'eta_jet', 'N_PFO', 'W_PFO_jet', 'C1_PFO_jet']
        variable_importances['variable'] = variable_importances['variable'].apply(
            lambda x: variables_names[int(x.split('.')[-1])])
        data_info.plot_feature_importance(variable_importances, os.path.join(
            args.params.logdir, f'feature_bdt_score.png'))
    log.info("Done!")


if __name__ == "__main__":
    main()
