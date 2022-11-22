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
from src.models import basicFC, transformer, highway #,BDT
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

    # train_files = []
    # dev_files = []
    # test_files = []
    # folders = os.listdir(args.data.path)
    # folders.sort()
    # slices = args.data.JZ_slices if args.data.JZ_slices is not None else list(range(1, 13))
    # folders = [folders[slc-1] for slc in slices]
    # log.info(f"Folders used for training: {folders}")
    # for folder in folders:
    #     train_files.append([os.path.join(args.data.path, folder, 'train', file) for file in os.listdir(
    #         os.path.join(args.data.path, folder, 'train'))])
    #     dev_files.append([os.path.join(args.data.path, folder, 'dev', file) for file in os.listdir(
    #         os.path.join(args.data.path, folder, 'dev'))])
    #     test_files.append([os.path.join(args.data.path, folder, 'test', file) for file in os.listdir(
    #         os.path.join(args.data.path, folder, 'test'))])

    # if len(train_files) == 0:
    #     log.error("No data found!")
    #     raise FileNotFoundError("No data found!")

    # dev_size = int(args.dataset.take *
    #                 args.dataset.dev_size) if args.dataset.take is not None and args.dataset.dev_size is not None else None
    # test_size = int(
    #     args.dataset.take*args.dataset.test_size) if args.dataset.take is not None and args.dataset.test_size is not None else None
    # train = get_preprocessed_dataset(train_files, args_data=args.data,
    #                                     args_dataset=args.dataset, name="train", size=args.dataset.take)
    # dev = get_preprocessed_dataset(dev_files, args_data=args.data,
    #                                 args_dataset=args.dataset, name="dev", size=dev_size)
    # test = get_preprocessed_dataset(test_files, args_data=args.data,
    #                                 args_dataset=args.dataset, name="test", size=test_size)
    # else:
    #     with open(os.path.join(args.data.cached, "train") + '/element_spec', 'rb') as in_:
    #         es = pickle.load(in_)

    #     train, dev, test = [tf.data.experimental.load(os.path.join(
    #         args.data.cached, f"{name}"), es) for name in ["train", "dev", "test"]]
    #     if args.dataset.take is not None:
    #         dev_size = int(args.dataset.take *
    #                        args.dataset.dev_size) if args.dataset.take is not None and args.dataset.dev_size is not None else None
    #         test_size = int(
    #             args.dataset.take*args.dataset.test_size) if args.dataset.take is not None and args.dataset.test_size is not None else None
    #         train = train.take(args.dataset.take)
    #         dev = dev.take(dev_size)
    #         test = test.take(test_size)

    #     train, dev, test = [ds.batch(args.dataset.batch_size).prefetch(tf.data.AUTOTUNE)
    #                         for ds in [train, dev, test]]
    path = '/Users/samueljankovych/Documents/MFF/bakalarka/JIDENN/data/data2/JZ04_full'
    train, dev, test = get_preprocessed_dataset([path], args_data=args.data, test_size=0.2, dev_size=0.2)

    @tf.function
    def get_constituents(sample, label, weight):
        return tf.RaggedTensor.from_tensor(tf.stack([sample['perJetTuple'][var] for var in args.data.variables.perJetTuple], axis=-1)), label, weight
    
    @tf.function
    def setup_dataset(dataset):
        dataset = dataset.map(get_constituents)
        dataset = dataset.batch(args.dataset.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    
    train = train.apply(setup_dataset)
    dev = dev.apply(setup_dataset)
    test = test.apply(setup_dataset)

    # train = train.batch(args.dataset.batch_size).prefetch(tf.data.AUTOTUNE)
    # dev = dev.batch(args.dataset.batch_size).prefetch(tf.data.AUTOTUNE)
    # test = test.batch(args.dataset.batch_size).prefetch(tf.data.AUTOTUNE)
    # 198815 all


    if args.preprocess.draw_distribution is not None and args.preprocess.draw_distribution > 0:
        log.info(f"Drawing data distribution with {args.preprocess.draw_distribution} samples")
        for name, dataset in zip(["train", "dev", "test"], [train, dev, test]):
            dataset = dataset.unbatch().take(args.preprocess.draw_distribution)
            dir = os.path.join(args.params.logdir, 'dist', name)
            os.makedirs(dir, exist_ok=True)
            df = data_info.tf_dataset_to_pandas(
                dataset=dataset, var_names=args.data.variables.perJet+args.data.variables.perEvent)
            df['named_label'] = df['label'].replace({0: args.data.labels[0], 1: args.data.labels[1]})
            data_info.generate_data_distributions(df=df, folder=dir)

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

    history = model.fit(train, validation_data=dev, epochs=args.params.epochs, callbacks=callbacks)

    model_dir = os.path.join(args.params.logdir, 'model')
    log.info(f"Saving model to {model_dir}")
    model.save(model_dir)

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
