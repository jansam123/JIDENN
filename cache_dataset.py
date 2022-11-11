import tensorflow as tf
import numpy as np
import pandas as pd
import os
import logging
import hydra
from hydra.core.config_store import ConfigStore
import pickle
#
from src.data.get_dataset import get_preprocessed_dataset
from src.config import config
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

    train = get_preprocessed_dataset(train_files, args_data=args.data,
                                     name="train").shuffle(args.dataset.shuffle_buffer)
    dev = get_preprocessed_dataset(dev_files, args_data=args.data, name="dev").shuffle(args.dataset.shuffle_buffer)
    test = get_preprocessed_dataset(test_files, args_data=args.data, name="test").shuffle(args.dataset.shuffle_buffer)

    for tst, trn, dvc in zip(test.take(5), train.take(5), dev.take(5)):
        print(tst, trn, dvc)

    base_path = "/home/jankovys/JIDENN/data/filtered2"
    os.makedirs(base_path, exist_ok=True)
    for ds, ds_name in zip([train, dev, test], ["train", "dev", "test"]):
        tf.data.experimental.save(ds, os.path.join(base_path, ds_name))
        with open(os.path.join(base_path, ds_name) + '/element_spec', 'wb') as out_:  # also save the element_spec to disk for future loading
            pickle.dump(ds.element_spec, out_)


if __name__ == "__main__":
    main()
