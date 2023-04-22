import tensorflow as tf
import awkward as ak
import pandas as pd


def pandas_to_tensor(df: pd.Series) -> tf.RaggedTensor:
    levels = df.index.nlevels
    if levels == 1:
        return tf.constant(df.values)
    elif levels == 2:
        row_lengths = df.groupby(level=[0]).count()
        return tf.RaggedTensor.from_row_lengths(df.values, row_lengths.values, validate=False)
    else:
        max_level_group = list(range(levels-1))
        nested_row_lengths = [df.groupby(level=max_level_group).count()]
        for i in range(1, levels-1):
            nested_row_lengths.append(nested_row_lengths[-1].groupby(level=max_level_group[:-i]).count())
        return tf.RaggedTensor.from_nested_row_lengths(df.values, nested_row_lengths=nested_row_lengths[::-1], validate=True)

def awkward_to_tensor(array: ak.Array) -> tf.RaggedTensor:
    if array.ndim == 1:
        return tf.constant(array.to_list())
    elif array.ndim == 2:
        row_lengths = ak.num(array, axis=1).to_list()
        return tf.RaggedTensor.from_row_lengths(ak.flatten(array, axis=None).to_list(), row_lengths=row_lengths, validate=False)
    else:
        nested_row_lengths = [ak.flatten(ak.num(array, axis=ax), axis=None).to_list()
                                for ax in range(1, array.ndim)]
        return tf.RaggedTensor.from_nested_row_lengths(ak.flatten(
            array, axis=None).to_list(), nested_row_lengths=nested_row_lengths, validate=False)
