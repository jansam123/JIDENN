import tensorflow as tf
from typing import Union

from src.config import config_subclasses as cfg
from .utils.CutV2 import Cut
from .JIDENNDatasetV2 import JIDENNDataset, JIDENNVariables


def get_preprocessed_dataset(files: list[str],
                             args_data: cfg.Data) -> JIDENNDataset:
    @tf.function
    def resample_g_q(_: JIDENNVariables, label: int, w: float) -> int:
        return 0 if label == args_data.raw_gluon else 1

    @tf.function
    def label_mapping(x: int) -> int:
        if x == args_data.raw_gluon:
            return args_data.gluon
        else:
            return args_data.quark

    JZ_cuts = args_data.JZ_cut if args_data.JZ_cut is not None else [None]*len(files)

    datasets = []
    for jz_cut, jz_file in zip(JZ_cuts, files):

        jidenn_dataset = JIDENNDataset(variables=args_data.variables,
                                       target=args_data.target,
                                       weight=args_data.weight)
        jidenn_dataset = jidenn_dataset.load_dataset(jz_file)
        jidenn_dataset = jidenn_dataset.process(cut=Cut(jz_cut) & Cut(args_data.cut) if args_data.cut is not None else Cut(jz_cut))
        jidenn_dataset = jidenn_dataset.resample_by_label(resample_g_q, [0.5, 0.5])
        jidenn_dataset = jidenn_dataset.remap_labels(label_mapping)
        datasets.append(jidenn_dataset)

    return JIDENNDataset.combine(datasets, args_data.JZ_weights)
