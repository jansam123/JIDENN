from typing import Callable
from .JIDENNDatasetV2 import JIDENNDataset, JIDENNVariables
from src.config import config_subclasses as cfg
from src.data.preprocess import pipe
import tensorflow as tf
from src.data.utils.CutV2 import Cut


def get_preprocessed_dataset(files: list[str],
                             args_data: cfg.Data,
                             name: str,
                             args_dataset: cfg.Dataset | None = None,
                             size: int | None = None) -> tf.data.Dataset:

    @tf.function
    def resample_g_q(_: JIDENNVariables, label: int, w: float) -> int:
        return 0 if label == args_data.raw_gluon else 1

    JZ_cuts = args_data.JZ_cut if args_data.JZ_cut is not None else [None]*len(files)

    datasets = []
    for jz_cut, jz_file in zip(JZ_cuts, files):
        
        jidenn_dataset = JIDENNDataset(file=jz_file,
                                variables=args_data.variables,
                                target=args_data.target,
                                weight=args_data.weight,
                                cut=Cut(jz_cut) if jz_cut is not None else None)
        
        jidenn_dataset.resample_by_label(resample_g_q, [0.5, 0.5])
        datasets.append(jidenn_dataset.dataset)

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

    if args_dataset is not None:
        return dataset.apply(pipe(args_dataset=args_dataset, name=name, take=size))
    elif size is not None:
        return dataset.take(size)
    else:
        return dataset
