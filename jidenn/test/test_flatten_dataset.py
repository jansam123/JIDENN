import pytest
import tensorflow as tf
from jidenn.preprocess.flatten_dataset import get_filter_ragged_values_fn, get_ragged_to_dataset_fn, get_filter_empty_fn


def test_get_filter_ragged_values_fn():
    filter_fn = get_filter_ragged_values_fn()
    sample = {
        'jets_PartonTruthLabelID': tf.constant([1, -1, 2, 3, -999]),
        'jets_pt': tf.constant([1, 2, 3, 4, 5]),
        'HLT': tf.constant(10),
        'vertex_id': tf.constant([1, 2, 3, 4]),
        'ragged_vertex_id': tf.ragged.constant([[1, 2], [1, 2, 3, 4], [1, 2, 3]]),
        'jets_const_pt': tf.ragged.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]),
    }
    expected = {
        'jets_PartonTruthLabelID': tf.constant([1, 2, 3]),
        'jets_pt': tf.constant([1, 3, 4]),
        'HLT': tf.constant(10),
        'vertex_id': tf.constant([1, 2, 3, 4]),
        'ragged_vertex_id': tf.ragged.constant([[1, 2], [1, 2, 3, 4], [1, 2, 3]]),
        'jets_const_pt': tf.ragged.constant([[1, 2, 3], [7, 8, 9], [10, 11, 12]]),
    }

    filtered_sample = filter_fn(sample)
    assert filtered_sample.keys() == expected.keys(), "Keys are not the same"
    for key, item in filtered_sample.items():
        if tf.rank(item) == 0:
            assert item == expected[key], f"Item {key} is not the same"
        else:
            assert tf.reduce_all(tf.equal(item, expected[key])), f"Item {key} is not the same"


def test_get_ragged_to_dataset_fn():
    flattner_fn = get_ragged_to_dataset_fn()
    sample = {
        'jets_PartonTruthLabelID': tf.constant([1, 2, 3]),
        'jets_pt': tf.constant([1, 3, 4]),
        'jets_n': tf.constant(3),
        'HLT': tf.constant(10),
        'vertex_id': tf.constant([1, 2, 3, 4]),
        'ragged_vertex_id': tf.ragged.constant([[1, 2], [1, 2, 3, 4], [1, 2, 3]]),
        'jets_const_pt': tf.ragged.constant([[1, 2, 3], [7, 8, 9], [10, 11, 12]]),
    }
    expected = [
        {
            'jets_PartonTruthLabelID': tf.constant(1),
            'jets_pt': tf.constant(1),
            'jets_n': tf.constant(3),
            'HLT': tf.constant(10),
            'vertex_id': tf.constant([1, 2, 3, 4]),
            'ragged_vertex_id': tf.ragged.constant([[1, 2], [1, 2, 3, 4], [1, 2, 3]]),
            'jets_const_pt': tf.constant([1, 2, 3]),
        },
        {
            'jets_PartonTruthLabelID': tf.constant(2),
            'jets_pt': tf.constant(3),
            'jets_n': tf.constant(3),
            'HLT': tf.constant(10),
            'vertex_id': tf.constant([1, 2, 3, 4]),
            'ragged_vertex_id': tf.ragged.constant([[1, 2], [1, 2, 3, 4], [1, 2, 3]]),
            'jets_const_pt': tf.constant([7, 8, 9]),
        },
        {
            'jets_PartonTruthLabelID': tf.constant(3),
            'jets_pt': tf.constant(4),
            'jets_n': tf.constant(3),
            'HLT': tf.constant(10),
            'vertex_id': tf.constant([1, 2, 3, 4]),
            'ragged_vertex_id': tf.ragged.constant([[1, 2], [1, 2, 3, 4], [1, 2, 3]]),
            'jets_const_pt': tf.constant([10, 11, 12]),
        },
    ]
    flat_sample = flattner_fn(sample)
    
    for calc, exp in zip(flat_sample, expected):
        assert calc.keys() == exp.keys(), "Keys are not the same"
        for key, item in calc.items():
            if tf.rank(item) == 0:
                assert item == exp[key], f"Item {key} is not the same"
            else:
                assert tf.reduce_all(tf.equal(item, exp[key])), f"Item {key} is not the same"

@pytest.mark.parametrize("sample, expected", [
    (
        {
        'jets_PartonTruthLabelID': tf.constant([1, 2, 3]),
        'jets_pt': tf.constant([1, 3, 4]),
        'jets_n': tf.constant(3),
        'HLT': tf.constant(10),
        'vertex_id': tf.constant([1, 2, 3, 4]),
        'ragged_vertex_id': tf.ragged.constant([[1, 2], [1, 2, 3, 4], [1, 2, 3]]),
        'jets_const_pt': tf.ragged.constant([[1, 2, 3], [7, 8, 9], [10, 11, 12]]),
    }, True),
    ({
        'jets_PartonTruthLabelID': tf.constant([]),
        'jets_pt': tf.constant([]),
        'jets_n': tf.constant(3),
        'HLT': tf.constant(10),
        'vertex_id': tf.constant([1, 2, 3, 4]),
        'ragged_vertex_id': tf.ragged.constant([[1, 2], [1, 2, 3, 4], [1, 2, 3]]),
        'jets_const_pt': tf.ragged.constant([]),
    }, False),
])           
def test_get_filter_empty_fn(sample, expected):
    filter_fn = get_filter_empty_fn()
    assert filter_fn(sample) == expected