import tensorflow as tf
from typing import Union

from src.config import config_subclasses as cfg
from src.data.preprocess import pipe
from .utils.CutV2 import Cut
from .JIDENNDatasetV2 import JIDENNDataset, JIDENNVariables


def process_JZ_datasets(datasets,
                        args_data: cfg.Data) -> tf.data.Dataset:
    if len(datasets) > 1:
        dataset = tf.data.Dataset.sample_from_datasets(datasets, weights=args_data.JZ_weights)
    else:
        dataset = datasets[0]

    @tf.function
    def label_mapping(x):
        if x == args_data.raw_gluon:
            return args_data.gluon
        else:
            return args_data.quark

    dataset = dataset.map(lambda x, y, z: (x, label_mapping(y), z), num_parallel_calls=tf.data.AUTOTUNE)
    return dataset


def get_JZ_datasets(files: list[str],
                    args_data: cfg.Data,
                    test_size: Union[float, None] = None,
                    dev_size: Union[float, None] = None) -> tuple[list[tf.data.Dataset], list[tf.data.Dataset], list[tf.data.Dataset]]:
    @tf.function
    def resample_g_q(_: JIDENNVariables, label: int, w: float) -> int:
        return 0 if label == args_data.raw_gluon else 1

    JZ_cuts = args_data.JZ_cut if args_data.JZ_cut is not None else [None]*len(files)

    train_datasets = []
    dev_datasets = []
    test_datasets = []
    for jz_cut, jz_file in zip(JZ_cuts, files):

        jidenn_dataset = JIDENNDataset(variables=args_data.variables,
                                       target=args_data.target,
                                       weight=args_data.weight)
        jidenn_dataset.load(jz_file)
        if dev_size is not None and test_size is not None:
            jidenn_dataset.split(test_size=test_size, dev_size=dev_size)
        jidenn_dataset.process(cut=Cut(jz_cut) if jz_cut is not None else None)
        jidenn_dataset.resample_by_label(resample_g_q, [0.5, 0.5])
        train_datasets.append(jidenn_dataset.train)
        dev_datasets.append(jidenn_dataset.dev)
        test_datasets.append(jidenn_dataset.test)
    return train_datasets, dev_datasets, test_datasets


def get_preprocessed_dataset(files: list[str],
                             args_data: cfg.Data,
                             test_size: Union[float, None]=None,
                             dev_size: Union[float, None] = None) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:

    trains, devs, tests = get_JZ_datasets(files, args_data, test_size, dev_size)
    train, dev, test = [process_JZ_datasets(ds, args_data) for ds in [trains, devs, tests]]
    return train, dev, test
