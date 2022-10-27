import src.data.data_info as data_info
from src.callbacks.get_callbacks import get_callbacks
from src.config import config
from src.postprocess.pipeline import postprocess_pipe
from src.models import basicFC, transformer, BDT, highway
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import logging
import hydra
from hydra.core.config_store import ConfigStore
from src.data.get_dataset import get_preprocessed_dataset
# os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


cs = ConfigStore.instance()
cs.store(name="args", node=config.JIDENNConfig)


@hydra.main(version_base="1.2", config_path="src/config", config_name="config")
def main(args: config.JIDENNConfig) -> None:
    log = logging.getLogger(__name__)
    args.data.input_size = len(args.data.variables.perJet) if args.data.variables.perJet is not None else 0
    args.data.input_size += len(args.data.variables.perEvent) if args.data.variables.perEvent is not None else 0
    args.data.input_size += len(args.data.variables.perJetTuple) if args.data.variables.perJetTuple is not None else 0

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

    # dataset preparation
    train_files = []
    dev_files = []
    test_files = []
    folders = os.listdir(args.data.path)
    folders.sort()
    slices = args.data.JZ_slices if args.data.JZ_slices is not None else list(range(1, 13))
    folders = [folders[slc-1] for slc in slices]
    log.info(f"Folders used for training: {folders}")
    for folder in folders:
        train_files.append([os.path.join(args.data.path, folder, 'train', file) for file in os.listdir(
            os.path.join(args.data.path, folder, 'train'))])
        dev_files.append([os.path.join(args.data.path, folder, 'dev', file) for file in os.listdir(
            os.path.join(args.data.path, folder, 'dev'))])
        test_files.append([os.path.join(args.data.path, folder, 'test', file) for file in os.listdir(
            os.path.join(args.data.path, folder, 'test'))])

    if len(train_files) == 0:
        log.error("No data found!")
        raise FileNotFoundError("No data found!")

    dev_size = int(args.dataset.take *
                   args.dataset.dev_size) if args.dataset.take is not None and args.dataset.dev_size is not None else None
    test_size = int(
        args.dataset.take*args.dataset.test_size) if args.dataset.take is not None and args.dataset.test_size is not None else None
    train = get_preprocessed_dataset(train_files, args_data=args.data,
                                     args_dataset=args.dataset, name="train", size=args.dataset.take)
    dev = get_preprocessed_dataset(dev_files, args_data=args.data, args_dataset=args.dataset, name="dev", size=dev_size)
    test = get_preprocessed_dataset(test_files, args_data=args.data,
                                    args_dataset=args.dataset, name="test", size=test_size)

    if args.preprocess.draw_distribution is not None:
        log.info(f"Drawing data distribution with {args.preprocess.draw_distribution} samples")
        for name, dataset in zip(["train", "dev", "test"], [train, dev, test]):
            dataset = dataset.unbatch().take(args.preprocess.draw_distribution)
            dir = os.path.join(args.params.logdir, 'dist', name)
            os.makedirs(dir, exist_ok=True)
            data_info.generate_data_distributions(dataset=dataset,
                                                  folder=dir,
                                                  var_names=args.data.variables.perJet+args.data.variables.perEvent)

    def _model():
        if args.preprocess.normalize and args.params.model != 'BDT':
            if args.preprocess.min_max_path is not None:
                min_max = pd.read_csv(args.preprocess.min_max_path, index_col=0)
                mins = []
                maxs = []
                for var in args.data.variables.perJet+args.data.variables.perEvent:
                    var = var.split('[')[0]
                    mins.append(min_max.loc[var]['min'])
                    maxs.append(min_max.loc[var]['max'])
                mean = tf.constant(mins)
                variance = (tf.constant(maxs) - tf.constant(mins))**2
                log.info(
                    f"Using loaded mins: {mins} and maxs: {maxs} for normalization with mean=mins and variance=(maxs-mins)**2")
                normalizer = tf.keras.layers.Normalization(axis=-1, mean=mean, variance=variance)
            else:
                normalizer = tf.keras.layers.Normalization(axis=-1)
                log.info("Getting std and mean of the dataset...")
                log.info(f"Subsample size: {args.preprocess.normalization_size}")
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

        elif args.params.model == 'BDT':
            model = BDT.create(args.bdt)

        else:
            raise NotImplementedError("Model not implemented")

        return model

    # creating model
    if len(gpus) < 2:
        model = _model()
    else:
        # mirrored_strategy = tf.distribute.MirroredStrategy()
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            model = _model()

    # callbacks
    callbacks = get_callbacks(args.params, log, dev)

    # running training
    # tf.keras.utils.plot_model(model, f"{args.params.logdir}/model.png", show_shapes=True, expand_nested=True)
    model.fit(train, validation_data=dev, epochs=args.params.epochs, callbacks=callbacks)
    
    # saving model
    model_dir = os.path.join(args.params.logdir, 'model')
    log.info(f"Saving model to {model_dir}")
    model.save(model_dir)

    if test is None:
        log.warning("No test dataset, skipping evaluation.")
        log.info("Done!")
        return

    print(model.evaluate(test, return_dict=True))

    @tf.function
    def labels_only(x, y, z):
        return y
    test_dataset_labels = test.unbatch().map(labels_only)
    postprocess_pipe(model, test, test_dataset_labels, args.params.logdir, log)

    log.info("Done!")


if __name__ == "__main__":
    main()
