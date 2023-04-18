import tensorflow as tf
from typing import List, Optional, Tuple, Callable

from jidenn.config import config_subclasses as cfg
from .utils.Cut import Cut
from .JIDENNDataset import JIDENNDataset, JIDENNVariables


def get_preprocessed_dataset(files: List[str],
                             args_data: cfg.Data,
                             files_labels: Optional[List[int]] = None,
                             resample_labels: bool = True) -> JIDENNDataset:
    """Loads and preprocesses a dataset from a list of files.

    Args:
        files: A list of strings containing the file paths of the datasets to be loaded. Each file path points to a dataset saved with tf.data.experimental.save.
        args_data: Data configuration containing the required preprocessing settings.
        files_labels: A list of strings containing the stamps of the datasets to be loaded. Each stamp is used to identify the datafile it belongs to.

    Returns:
        A JIDENNDataset object containing the preprocessed dataset.
    """

    raw_quarks = tf.constant(args_data.raw_quarks, dtype=tf.int32)

    @tf.function
    def resample_g_q(_: JIDENNVariables, x: int, w: float) -> int:
        if tf.equal(x, args_data.raw_gluon):
            return 0
        elif tf.reduce_any(tf.equal(x, raw_quarks)):
            return 1
        else:
            return -999

    @tf.function
    def label_mapping(x: int) -> int:
        if tf.equal(x, args_data.raw_gluon):
            return args_data.gluon
        elif tf.reduce_any(tf.equal(x, raw_quarks)):
            return args_data.quark
        else:
            return -999

    def stamp_origin_file(stamp: int) -> Callable[[JIDENNVariables], JIDENNVariables]:
        @tf.function
        def stamp_origin_file_wrap(data: JIDENNVariables) -> JIDENNVariables:
            new_data = data.copy()
            per_event = new_data['perEvent'].copy()
            per_event['origin_file'] = stamp
            new_data['perEvent'] = per_event
            return new_data
        return stamp_origin_file_wrap

    JZ_cuts = args_data.JZ_cut if args_data.JZ_cut is not None else [None] * len(files)

    datasets = []
    for i, (jz_cut, jz_file) in enumerate(zip(JZ_cuts, files)):

        jidenn_dataset = JIDENNDataset(variables=args_data.variables,
                                       target=args_data.target,
                                       weight=args_data.weight)
        jidenn_dataset = jidenn_dataset.load_dataset(jz_file)

        if jz_cut is not None:
            cut = Cut(jz_cut) & Cut(args_data.cut) if args_data.cut is not None else Cut(jz_cut)
        elif args_data.cut is not None:
            cut = Cut(args_data.cut)
        else:
            cut = None

        jidenn_dataset = jidenn_dataset.process(cut=cut) if cut is not None else jidenn_dataset

        jidenn_dataset = jidenn_dataset.resample_dataset(
            resample_g_q, [0.5, 0.5]) if resample_labels else jidenn_dataset
        jidenn_dataset = jidenn_dataset.map_data(stamp_origin_file(
            files_labels[i])) if files_labels is not None else jidenn_dataset

        jidenn_dataset = jidenn_dataset.remap_labels(label_mapping)
        datasets.append(jidenn_dataset)

    if len(datasets) == 1:
        return datasets[0]

    return JIDENNDataset.combine(datasets, args_data.JZ_weights)


def cache_dataset(jidenn_dataset: JIDENNDataset, size: int, path: str, num_shards: int = 10):

    def _prep(dataset: tf.data.Dataset) -> tf.data.Dataset:
        dataset = dataset.take(size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    jidenn_dataset = jidenn_dataset.apply(_prep)
    jidenn_dataset.save_dataset(path, num_shards)
