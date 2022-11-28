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
from src.postprocess.tb_plots import tb_postprocess
from src.postprocess.pipeline import postprocess_pipe
from src.models import basicFC, transformer, highway, BDT, part
from src.data.JIDENNDatasetV2 import get_constituents, get_high_level_variables
# os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


cs = ConfigStore.instance()
cs.store(name="args", node=config.JIDENNConfig)


@hydra.main(version_base="1.2", config_path="src/config", config_name="config")
def main(args: config.JIDENNConfig) -> None:
    log = logging.getLogger(__name__)
    # args.data.input_size = len(args.data.variables.perJet) if args.data.variables.perJet is not None else 0
    # args.data.input_size += len(args.data.variables.perEvent) if args.data.variables.perEvent is not None else 0
    # args.data.input_size += len(args.data.variables.perJetTuple) if args.data.variables.perJetTuple is not None else 0

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
        tf.data.experimental.enable_debug_mode()
        tf.config.run_functions_eagerly(True)

    # fixing seed for reproducibility
    if args.params.seed is not None:
        log.info(f"Setting seed to {args.params.seed}")
        np.random.seed(args.params.seed)
        tf.random.set_seed(args.params.seed)

    # managing threads
    if args.params.threads is not None:
        tf.config.threading.set_inter_op_parallelism_threads(args.params.threads)
        tf.config.threading.set_intra_op_parallelism_threads(args.params.threads)

    mirrored_strategy = tf.distribute.MirroredStrategy() if len(gpus) > 1 else None

    files = []
    for name in ["train", "test", "dev"]:
        file = [f'{args.data.path}/{file}' for file in args.data.JZ_slices] if args.data.JZ_slices is not None else [
            f'{args.data.path}/{file}/{name}' for file in os.listdir(args.data.path)]
        files.append(file)

    train, test, dev = [get_preprocessed_dataset(file, args.data) for file in files]

    if args.preprocess.draw_distribution is not None and args.preprocess.draw_distribution > 0:
        log.info(f"Drawing data distribution with {args.preprocess.draw_distribution} samples")
        for name, dataset in zip(["train"], [train]):
            dataset = dataset.apply(lambda x: x.take(args.preprocess.draw_distribution))
            dir = os.path.join(args.params.logdir, 'dist', name)
            os.makedirs(dir, exist_ok=True)
            df = dataset.to_pandas()
            if 'perJetTuple.jets_PFO_pt' in df.columns:
                df['perJet.jets_PFO_n'] = df['perJetTuple.jets_PFO_pt'].apply(lambda x: len(x))
            df['named_label'] = df['label'].replace({0: args.data.labels[0], 1: args.data.labels[1]})
            data_info.generate_data_distributions(df=df, folder=dir)

    if args.dataset.take is not None:
        train_size = 1 - args.dataset.dev_size + args.dataset.test_size
        train_size = int(train_size * args.dataset.take)
        dev_size = int(args.dataset.dev_size * args.dataset.take)
        test_size = int(args.dataset.test_size * args.dataset.take)
    else:
        train_size, dev_size, test_size = None, None, None

    model_to_data_mapping = {
        "basic_fc": get_high_level_variables,
        "transformer": get_constituents,
        "highway": get_high_level_variables,
        "BDT": get_high_level_variables,
        "part": get_constituents
    }

    train = train.get_dataset(batch_size=args.dataset.batch_size,
                              shuffle_buffer_size=args.dataset.shuffle_buffer,
                              map_func=model_to_data_mapping[args.params.model],
                              take=train_size,
                              assert_shape=True)

    dev = dev.get_dataset(batch_size=args.dataset.batch_size,
                          map_func=model_to_data_mapping[args.params.model],
                          take=dev_size)

    test = test.get_dataset(batch_size=args.dataset.batch_size,
                            map_func=model_to_data_mapping[args.params.model],
                            take=test_size)

    def _model():
        if args.preprocess.normalize and args.params.model != 'BDT':
            normalizer = tf.keras.layers.Normalization(axis=-1)
            log.info("Getting std and mean of the dataset...")
            log.info(f"Subsample size: {args.preprocess.normalization_size}")
            try:
                normalizer.adapt(train.map(lambda x, y, z: x.to_tensor()), steps=args.preprocess.normalization_size)
            except AttributeError:
                normalizer.adapt(train.map(lambda x, y, z: x), steps=args.preprocess.normalization_size)
        else:
            normalizer = None
            log.warning("Normalization disabled.")

        if args.params.model == "basic_fc":
            model = basicFC.create(args.params, args.basic_fc, args.data, preprocess=normalizer)
            model.summary(print_fn=log.info)

        elif args.params.model == "highway":
            model = highway.create(args.params, args.highway, args.data, preprocess=normalizer)
            model.summary(print_fn=log.info)

        elif args.params.model == "transformer":
            model = transformer.create(args.params, args.transformer, args.data, preprocess=normalizer)
            model.summary(print_fn=log.info)

        elif args.params.model == "part":
            model = part.create(args.params, args.part, args.data, preprocess=normalizer)
            model.summary(print_fn=log.info)

        elif args.params.model == 'BDT':
            args.params.epochs = 1
            model = BDT.create(args.bdt)

        else:
            raise NotImplementedError("Model not implemented")

        return model

    # creating model
    if len(gpus) < 2:
        model = _model()
    else:
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            model = _model()

    # callbacks
    callbacks = get_callbacks(args.params, log, dev)

    # running training
    history = model.fit(train, epochs=args.params.epochs, callbacks=callbacks, validation_data=dev, verbose=1)

    model_dir = os.path.join(args.params.logdir, 'model')
    log.info(f"Saving model to {model_dir}")
    model.save(model_dir, save_format='tf')

    if args.params.model != 'BDT':
        log.info(f"Saving history")
        history_dir = os.path.join(args.params.logdir, 'history')
        os.makedirs(history_dir, exist_ok=True)
        for metric in [m for m in history.history.keys() if 'val' not in m]:
            tb_postprocess(
                {f'{metric}': history.history[metric], f'validation {metric}': history.history[f'val_{metric}']}, history_dir, metric, args.params.epochs)

    # saving model

    if test is None:
        log.warning("No test dataset, skipping evaluation.")
        log.info("Done!")
        return

    print(model.evaluate(test, return_dict=True))

    if args.params.model == 'BDT':
        model.summary(print_fn=log.info)
        variable_importance_metric = "SUM_SCORE"
        variable_importances = model.make_inspector().variable_importances()[variable_importance_metric]
        variable_importances = pd.DataFrame({'variable': [str(vi[0].name) for vi in variable_importances],
                                             'score': [vi[1] for vi in variable_importances]})
        all_variables = args.data.variables.perJet + args.data.variables.perEvent
        variable_importances['variable'] = variable_importances['variable'].apply(
            lambda x: all_variables[int(x.split('.')[-1])])
        print(variable_importances)
        data_info.plot_feature_importance(variable_importances, os.path.join(
            args.params.logdir, f'feature_bdt_score.png'))

        df = data_info.tf_dataset_to_pandas(dataset=test.unbatch(
        ), var_names=args.data.variables.perJet+args.data.variables.perEvent)
        score = model.predict(test).ravel()
        df['score'] = score
        df['prediction'] = score.round()
        df['named_label'] = df['label'].replace({0: args.data.labels[0], 1: args.data.labels[1]})
        df['named_prediction'] = df['prediction'].replace({0: args.data.labels[0], 1: args.data.labels[1]})
        postprocess_pipe(df, args.params.logdir, log=log)

    log.info("Done!")


if __name__ == "__main__":
    main()
