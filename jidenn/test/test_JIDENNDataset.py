import pytest
import tempfile
import os
import numpy as np
import tensorflow as tf
import logging
import pandas as pd
logging.basicConfig(level=logging.INFO)

from jidenn.data.JIDENNDataset import JIDENNDataset, get_stacker, FLOAT_PRECISION


@pytest.fixture
def node_data_small():
    # Create ROOTVariables with two variables.
    # Each variable is a tensor of shape [N] with N=5.
    return {
        'var1': tf.constant(np.arange(5), dtype=tf.float32),
        'var2': tf.constant(np.arange(100, 105), dtype=tf.float32)
    }
    
@pytest.fixture
def highlevel_data_small():
    # Create ROOTVariables with two variables.
    # Each variable is a scalar.
    return {
        'var1': tf.constant(5, dtype=tf.float32),
        'var2': tf.constant(100, dtype=tf.float32)
    }
    
@pytest.fixture
def node_data_exact():
    # Create ROOTVariables where the number of constituents equals max_constituents.
    return {
        'var1': tf.constant(np.arange(10), dtype=tf.float32),
        'var2': tf.constant(np.arange(200, 210), dtype=tf.float32)
    }

@pytest.fixture
def edge_data_small():
    # Create ROOTVariables with two edge variables.
    # Each variable is a tensor of shape [N, N] with N=3.
    return {
        'edge1': tf.constant(np.arange(9).reshape(3, 3), dtype=tf.float32),
        'edge2': tf.constant(np.arange(100, 109).reshape(3, 3), dtype=tf.float32)
    }

def test_node_padding(node_data_small):
    max_constituents = 10
    pad_value = -999.0
    stacker = get_stacker(max_constituents, pad_value)
    
    # Call the function. For node variables (rank 2) we now expect a tuple (tensor, mask)
    result_tensor, result_mask = stacker(node_data_small)
    # Expected padded tensor shape: [max_constituents, num_vars] = [10, 2]
    assert result_tensor.shape == (max_constituents, 2)
    # Expected mask shape: [max_constituents,]
    assert result_mask.shape == (max_constituents,)
    
    # Convert to numpy for easier testing.
    result_tensor_np = result_tensor.numpy()
    result_mask_np = result_mask.numpy()
    # The first 5 rows should match the stacked values.
    expected_stacked = np.stack([
        np.array(node_data_small['var1']),
        np.array(node_data_small['var2'])
    ], axis=-1)
    np.testing.assert_array_equal(result_tensor_np[:5], expected_stacked)
    # The padded rows (rows 5 to 9) should be filled with pad_value.
    np.testing.assert_array_equal(result_tensor_np[5:], pad_value * np.ones((max_constituents - 5, 2), dtype=np.float32))
    # Check mask: first 5 True, rest False.
    expected_mask = np.concatenate([np.ones(5, dtype=bool), np.zeros(max_constituents - 5, dtype=bool)])
    np.testing.assert_array_equal(result_mask_np, expected_mask)

def test_node_exact(node_data_exact):
    max_constituents = 10
    pad_value = -999.0
    stacker = get_stacker(max_constituents, pad_value)
    
    result_tensor, result_mask = stacker(node_data_exact)
    assert result_tensor.shape == (max_constituents, 2)
    assert result_mask.shape == (max_constituents,)
    
    result_tensor_np = result_tensor.numpy()
    # Since the number of constituents equals max_constituents, no padding is added.
    expected_stacked = np.stack([
        np.array(node_data_exact['var1']),
        np.array(node_data_exact['var2'])
    ], axis=-1)
    np.testing.assert_array_equal(result_tensor_np, expected_stacked)
    # And the mask should be all True.
    expected_mask = np.ones(max_constituents, dtype=bool)
    np.testing.assert_array_equal(result_mask.numpy(), expected_mask)

def test_edge_padding(edge_data_small):
    max_constituents = 5
    pad_value = -999.0
    stacker = get_stacker(max_constituents, pad_value)
    
    result_tensor, result_mask = stacker(edge_data_small)
    # Expected shape: [max_constituents, max_constituents, num_vars] = [5, 5, 2]
    assert result_tensor.shape == (max_constituents, max_constituents, 2)
    assert result_mask.shape == (max_constituents,)
    
    result_np = result_tensor.numpy()
    # First 3 rows and columns should match the stacked edge values.
    expected_edge1 = np.array(edge_data_small['edge1'])
    expected_edge2 = np.array(edge_data_small['edge2'])
    expected_stacked = np.stack([expected_edge1, expected_edge2], axis=-1)
    np.testing.assert_array_equal(result_np[:3, :3], expected_stacked)
    # The padded areas (rows or columns >= 3) should be pad_value.
    pad_area = pad_value * np.ones((max_constituents, max_constituents, 2), dtype=np.float32)
    pad_area[:3, :3] = expected_stacked
    np.testing.assert_array_equal(result_np, pad_area)
    # Check mask: first 3 True, rest False.
    expected_mask = np.concatenate([np.ones(3, dtype=bool), np.zeros(max_constituents - 3, dtype=bool)])
    np.testing.assert_array_equal(result_mask.numpy(), expected_mask)

def test_tuple_input(node_data_small, node_data_exact):
    max_constituents = 10
    pad_value = -999.0
    stacker = get_stacker(max_constituents, pad_value)
    
    # Provide a tuple of ROOTVariables.
    # In the original function a tuple input produced a tuple (node_tensor, edge_tensor)
    # Here we assume the first dictionary is for nodes and the second for edges.
    result = stacker((node_data_small, node_data_exact))
    # Now result should be a tuple of three elements: (node_tensor, edge_tensor, node_mask)
    assert isinstance(result, tuple)
    assert len(result) == 3
    node_tensor, edge_tensor, node_mask = result
    # First tensor (from node_data_small) should be padded.
    assert node_tensor.shape == (max_constituents, 2)
    # Second tensor (from node_data_exact) should have shape (max_constituents, 2)
    assert edge_tensor.shape == (max_constituents, 2)
    # The mask should have shape (max_constituents,)
    assert node_mask.shape == (max_constituents,)
    
    # Check that the first tensor is padded correctly.
    node_tensor_np = node_tensor.numpy()
    expected0 = np.stack([
        np.array(node_data_small['var1']),
        np.array(node_data_small['var2'])
    ], axis=-1)
    np.testing.assert_array_equal(node_tensor_np[:5], expected0)
    np.testing.assert_array_equal(node_tensor_np[5:], pad_value * np.ones((max_constituents - 5, 2), dtype=np.float32))
    # And check the mask.
    expected_mask = np.concatenate([np.ones(5, dtype=bool), np.zeros(max_constituents - 5, dtype=bool)])
    np.testing.assert_array_equal(node_mask.numpy(), expected_mask)

def test_highlevel_stacking(highlevel_data_small):
    # For high-level variables the inputs are scalars, so stacking gives a rank-1 tensor.
    stacker = get_stacker()
    
    result = stacker(highlevel_data_small)
    # Since the input tensors are scalars, the stacked result is a rank-1 tensor.
    # In that case no mask is added.
    assert isinstance(result, tf.Tensor)
    assert result.shape == (2,)
    
    result_np = result.numpy()
    expected_stacked = np.stack([
        np.array(highlevel_data_small['var1']),
        np.array(highlevel_data_small['var2'])
    ], axis=-1)
    np.testing.assert_array_equal(result_np, expected_stacked)
    

num_events = 100


@pytest.fixture
def np_data_sample():
    return {
        'eventNumber': tf.range(num_events),
        'feature1': tf.random.uniform((num_events,)),
        'feature2': tf.random.uniform((num_events,)),
        'label': tf.random.uniform((num_events,), maxval=10, dtype=tf.int32),
        'weight': tf.random.uniform((num_events,), maxval=1),
    }


@pytest.fixture
def data_sample():
    # Create a list of NumPy arrays with random lengths
    feature1_list = [np.random.rand(np.random.randint(1, 10)) for _ in range(num_events)]
    feature2_list = [np.random.rand(np.random.randint(1, 10)) for _ in range(num_events)]

    # Convert the list of NumPy arrays to a ragged tensor
    feature1_ragged = tf.ragged.constant(feature1_list)
    feature2_ragged = tf.ragged.constant(feature2_list)

    # Create a dictionary with the sample data
    return {
        'eventNumber': tf.range(num_events),
        'feature1': feature1_ragged,
        'feature2': feature2_ragged,
        'label': tf.random.uniform((num_events,), maxval=10, dtype=tf.int32),
        'weight': tf.random.uniform((num_events,), maxval=1),
    }


@pytest.fixture
def jidenn_dataset(data_sample):
    dataset = tf.data.Dataset.from_tensor_slices(data_sample)
    element_spec = dataset.element_spec
    metadata = {'num_events': num_events}
    return JIDENNDataset(dataset,
                         element_spec=element_spec,
                         metadata=metadata,
                         length=num_events)


@pytest.fixture
def np_jidenn_dataset(np_data_sample):
    dataset = tf.data.Dataset.from_tensor_slices(np_data_sample)
    element_spec = dataset.element_spec
    metadata = {'num_events': num_events}
    return JIDENNDataset(dataset,
                         element_spec=element_spec,
                         metadata=metadata,
                         length=num_events)


def test_from_tf_dataset(jidenn_dataset, data_sample):
    dataset = tf.data.Dataset.from_tensor_slices(data_sample)
    metadata = {'num_events': num_events}
    new_dataset = JIDENNDataset.from_tf_dataset(dataset, metadata=metadata)

    assert new_dataset.metadata == jidenn_dataset.metadata, "Metadata is not the same"
    assert new_dataset.element_spec == jidenn_dataset.element_spec, "Element spec is not the same"
    assert new_dataset.length == jidenn_dataset.length, "Dataset cardinality is not the same"


def test_from_root_file():
    root_file = 'jidenn/test/test.root'
    dataset = JIDENNDataset.from_root_file(root_file, backend='ak')
    single_element = next(iter(dataset.dataset))

    assert dataset.dataset.cardinality().numpy() > 0, "Dataset is empty"
    assert dataset.metadata is not None, "Metadata is None"
    assert all([isinstance(x, tf.Tensor) for x in dataset.metadata.values()]), "Single element is not a tensor"
    assert dataset.dataset is not None, "Dataset is None"
    assert isinstance(single_element, dict), "Single element is not a dict"
    assert single_element.keys() == dataset.element_spec.keys(), "Keys of single element and element spec do not match"
    assert dataset.length == dataset.dataset.cardinality().numpy(), "Number of events is not correct"


def test_set_variables_target_weight(jidenn_dataset,):
    # Test setting the variables, target, and weight
    dataset = jidenn_dataset.set_variables_target_weight(
        variables=['feature1', 'feature2'], target='label', weight='weight')

    assert len(dataset.element_spec) == 3, "Number of elements in element spec is not 3"
    assert list(dataset.element_spec[0].keys()) == ['feature1', 'feature2'], "Keys of element spec do not match"

    dataset = jidenn_dataset.set_variables_target_weight(
        variables=['feature1', 'feature2', 'weight'], target='label', weight=None)

    assert len(dataset.element_spec) == 2, "Number of elements in element spec is not 2"
    assert list(dataset.element_spec[0].keys()) == ['feature1',
                                                    'feature2', 'weight'], "Keys of element spec do not match"
    assert dataset.length == num_events, "Number of events is not correct"


def test_remap_labels(jidenn_dataset,):
    # Test remapping the labels in the dataset
    def label_mapping(x): return x + 1
    remapped_dataset = jidenn_dataset.set_variables_target_weight(
        variables=['feature1', 'feature2'], target='label', weight='weight').remap_labels(label_mapping)

    # Test that the labels are remapped correctly
    for x, (re_x, re_y, re_z) in zip(jidenn_dataset.dataset.as_numpy_iterator(), remapped_dataset.dataset.as_numpy_iterator()):
        assert label_mapping(x['label']) == re_y, "Labels are not remapped correctly"
        assert tf.reduce_all(x['weight'] == re_z), "Weights are changed"
        assert tf.reduce_all(x['feature1'] == re_x['feature1']), "Feature1 is changed"
        assert tf.reduce_all(x['feature2'] == re_x['feature2']), "Feature2 is changed"

    assert remapped_dataset.length == num_events, "Number of events is not correct"


def test_remap_data(jidenn_dataset):
    # Test remapping the labels in the dataset
    def data_mapping(x): return {list(x.keys())[0]: x[list(x.keys())[0]
                                                      ] + 1, list(x.keys())[1]: x[list(x.keys())[1]] * 10}
    remapped_dataset = jidenn_dataset.set_variables_target_weight(
        variables=['eventNumber', 'feature1', 'feature2'], target='label', weight='weight').remap_data(data_mapping)

    # Test that the labels are remapped correctly
    for x, (re_x, re_y, re_z) in zip(jidenn_dataset.dataset.as_numpy_iterator(), remapped_dataset.dataset.as_numpy_iterator()):
        remaped_x = data_mapping(x)
        for key in remaped_x.keys():
            assert key in re_x.keys(), "Key is not in remapped data"
            assert tf.reduce_all(remaped_x[key] == re_x[key]), "Data is not remapped correctly"

    assert remapped_dataset.length == num_events, "Number of events is not correct"

    remapped_dataset = jidenn_dataset.remap_data(data_mapping)

    # Test that the labels are remapped correctly
    for x, re_x in zip(jidenn_dataset.dataset.as_numpy_iterator(), remapped_dataset.dataset.as_numpy_iterator()):
        remaped_x = data_mapping(x)
        for key in remaped_x.keys():
            assert key in re_x.keys(), "Key is not in remapped data"
            assert tf.reduce_all(remaped_x[key] == re_x[key]), "Data is not remapped correctly"

    assert remapped_dataset.length == num_events, "Number of events is not correct"


def test_combine(jidenn_dataset):
    # Test combining two datasets
    dataset_1 = jidenn_dataset.set_variables_target_weight(
        variables=['feature1', 'feature2'], target='label', weight='weight')
    dataset_2 = jidenn_dataset.set_variables_target_weight(
        variables=['feature1', 'feature2'], target='label', weight='weight')

    combined_dataset = JIDENNDataset.combine([dataset_1, dataset_2], mode='interleave')
    assert combined_dataset.metadata['num_events'] == 2 * \
        jidenn_dataset.length, "Number of events is not correct"
    assert combined_dataset.length == dataset_1.length + dataset_2.length, "Length is not correct"

    combined_dataset = JIDENNDataset.combine([dataset_1, dataset_2], mode='concatenate')
    assert combined_dataset.metadata['num_events'] == 2 * \
        jidenn_dataset.length, "Number of events is not correct"
    assert combined_dataset.length == dataset_1.length + dataset_2.length, "Length is not correct"

    # Test if ValueError is raised when combining different datasets
    dataset_3 = jidenn_dataset.set_variables_target_weight(
        variables=['feature2'], target='label', weight='weight')
    with pytest.raises(ValueError):
        JIDENNDataset.combine([dataset_1, dataset_3], mode='concatenate')
        JIDENNDataset.combine([dataset_1, dataset_3], mode='interleave')


def test_save_load_dataset(jidenn_dataset):
    # Test saving a dataset
    with tempfile.TemporaryDirectory() as tmp_dir:
        jidenn_dataset.save(os.path.join(tmp_dir, 'dataset'))
        jidenn_dataset.save(os.path.join(tmp_dir, 'dataset2'))
        assert os.path.exists(os.path.join(tmp_dir, 'dataset')), "Dataset directory does not exist"
        assert os.path.exists(os.path.join(tmp_dir, 'dataset', 'metadata.pkl')), "Metadata file does not exist"
        assert os.path.exists(os.path.join(tmp_dir, 'dataset', 'element_spec.pkl')
                              ), "Element spec file does not exist"

        loaded_dataset = JIDENNDataset.load(os.path.join(tmp_dir, 'dataset'))

        assert jidenn_dataset.metadata == loaded_dataset.metadata, "Metadata is not the same"
        assert jidenn_dataset.element_spec == loaded_dataset.element_spec, "Element spec is not the same"
        assert jidenn_dataset.length == loaded_dataset.length, "Dataset cardinality is not the same"

        loaded_dataset = JIDENNDataset.load([os.path.join(tmp_dir, 'dataset'), os.path.join(tmp_dir, 'dataset2')])

        for key, value in jidenn_dataset.metadata.items():
            assert 2*value == loaded_dataset.metadata[key], "Metadata is not the same"
        assert jidenn_dataset.element_spec == loaded_dataset.element_spec, "Element spec is not the same"
        assert 2*jidenn_dataset.length == loaded_dataset.length, "Dataset cardinality is not the same"
        
def test_load_parallel(jidenn_dataset):
    # Test saving a dataset
    with tempfile.TemporaryDirectory() as tmp_dir:
        jidenn_dataset.save(os.path.join(tmp_dir, 'dataset'))
        jidenn_dataset.save(os.path.join(tmp_dir, 'dataset2'))
        assert os.path.exists(os.path.join(tmp_dir, 'dataset')), "Dataset directory does not exist"
        assert os.path.exists(os.path.join(tmp_dir, 'dataset', 'metadata.pkl')), "Metadata file does not exist"
        assert os.path.exists(os.path.join(tmp_dir, 'dataset', 'element_spec.pkl')
                              ), "Element spec file does not exist"


        loaded_dataset = JIDENNDataset.load_parallel([os.path.join(tmp_dir, 'dataset'), os.path.join(tmp_dir, 'dataset2')])

        for key, value in jidenn_dataset.metadata.items():
            assert 2*value == loaded_dataset.metadata[key], "Metadata is not the same"
        assert jidenn_dataset.element_spec == loaded_dataset.element_spec, "Element spec is not the same"
        assert None == loaded_dataset.length, "Dataset cardinality is not the same"
        
        # loaded_dataset = JIDENNDataset.load_parallel([os.path.join(tmp_dir, 'dataset'), os.path.join(tmp_dir, 'dataset2')], 
        #                                              file_labels=['dataset', 'dataset2'])

        # for key, value in jidenn_dataset.metadata.items():
        #     assert 2*value == loaded_dataset.metadata[key], "Metadata is not the same"
        # assert jidenn_dataset.element_spec == loaded_dataset.element_spec, "Element spec is not the same"
        # assert None == loaded_dataset.length, "Dataset cardinality is not the same"


def test_apply(jidenn_dataset):
    # Test applying a function to the dataset
    def apply_fn(x):
        return x.take(10)

    applied_dataset = jidenn_dataset.apply(apply_fn)

    assert applied_dataset.length is None, "Dataset cardinality is not None"
    assert sum(1 for _ in applied_dataset.dataset) == 10, "Dataset size is not correct"
    assert jidenn_dataset.metadata == applied_dataset.metadata, "Metadata is not the same"
    assert jidenn_dataset.element_spec == applied_dataset.element_spec, "Element spec is not the same"


def test_filter(jidenn_dataset):
    # Test filtering the dataset
    def filter_fn(x):
        return x['label'] == 1

    filtered_dataset = jidenn_dataset.filter(filter_fn)

    assert filtered_dataset.length is None, "Dataset cardinality is not None"
    assert sum(1 for _ in filtered_dataset.dataset) == sum(
        1 for i in jidenn_dataset.dataset if filter_fn(i)), "Dataset size is not correct"
    assert jidenn_dataset.metadata == filtered_dataset.metadata, "Metadata is not the same"
    assert jidenn_dataset.element_spec == filtered_dataset.element_spec, "Element spec is not the same"


def test_to_pandas(jidenn_dataset, data_sample):
    # Test converting the dataset to a pandas dataframe

    try:
        df = jidenn_dataset.to_pandas()
    except ImportError:
        logging.warning(
            'Skipping test_to_pandas because tensorflow_datasets is not installed or is not functiing properly.')
        return
    assert df.equals(pd.DataFrame(data_sample)), "Dataframe is not the same"
    assert len(df) == jidenn_dataset.cardinality(
    ), "Dataframe size is not the same"
    assert set(df.columns) == set(
        data_sample.keys()), "Dataframe columns are not the same"


def test_to_numpy(jidenn_dataset, data_sample):
    # Test converting the dataset to a numpy array
    np_arrays = jidenn_dataset.to_numpy()
    for key in data_sample.keys():
        assert key in np_arrays.keys(), f"Key {key} is not in the numpy array"
        assert tf.reduce_all(np_arrays[key] == data_sample[key]), f"Elements of key {key} are not the same"


def test_plot_single_variable(np_jidenn_dataset):
    # Test plotting a single distribution
    with tempfile.TemporaryDirectory() as tmp_dir:
        np_jidenn_dataset.plot_single_variable(variable='feature1', save_path=os.path.join(
            tmp_dir, 'test.png'), hue_variable='label')
        assert os.path.exists(os.path.join(tmp_dir, 'test.png')), "Plot does not exist"


def test_get_prepared_dataset():
    rand_idx = np.random.randint(1, 10)
    feature1_list = [np.random.rand(rand_idx) for _ in range(num_events)]
    feature2_list = [np.random.rand(rand_idx) for _ in range(num_events)]

    # Convert the list of NumPy arrays to a ragged tensor
    feature1_ragged = tf.ragged.constant(feature1_list)
    feature2_ragged = tf.ragged.constant(feature2_list)

    # Create a dictionary with the sample data
    data_sample = {
        'feature1': feature1_ragged,
        'feature2': feature2_ragged,
    }
    dataset = tf.data.Dataset.from_tensor_slices(data_sample)
    element_spec = dataset.element_spec
    metadata = {'num_events': num_events}
    ds = JIDENNDataset(dataset,
                       element_spec=element_spec,
                       metadata=metadata,
                       length=num_events)
    batch_size = 10
    shuffle_buffer_size = 100
    take = 50

    prepared_dataset = ds.get_prepared_dataset(batch_size=batch_size,
                                               assert_length=True,
                                               shuffle_buffer_size=shuffle_buffer_size,
                                               ragged=True,
                                               take=take)
    assert isinstance(prepared_dataset.element_spec, tf.TensorSpec) or isinstance(
        prepared_dataset.element_spec, tf.RaggedTensorSpec), "Element spec is not a tensor spec or tuple"
    assert next(iter(prepared_dataset)).shape[0] == batch_size, "Batch size is not correct"
    assert prepared_dataset.cardinality().numpy() == take // batch_size, "Dataset cardinality is not correct"


def test_split_train_dev_test(jidenn_dataset):
    # Test splitting the dataset into train, dev and test
    train, dev, test = jidenn_dataset.split_train_dev_test(
        train_fraction=0.8, dev_fraction=0.1, test_fraction=0.1)

    assert train.metadata == jidenn_dataset.metadata, "Metadata is not the same"
    assert train.element_spec.keys() == jidenn_dataset.element_spec.keys(), "Element spec is not the same"

    assert dev.metadata == jidenn_dataset.metadata, "Metadata is not the same"
    assert dev.element_spec.keys() == jidenn_dataset.element_spec.keys(), "Element spec is not the same"

    assert test.metadata == jidenn_dataset.metadata, "Metadata is not the same"
    assert test.element_spec.keys() == jidenn_dataset.element_spec.keys(), "Element spec is not the same"

    assert train.length is None, "Dataset length must be unknown."
    assert dev.length is None, "Dataset length must be unknown."
    assert test.length is None, "Dataset length must be unknown."

    train_len = sum(1 for _ in train.dataset)
    dev_len = sum(1 for _ in dev.dataset)
    test_len = sum(1 for _ in test.dataset)
    print(f"Train: {train_len}, Dev: {dev_len}, Test: {test_len}")

    assert train_len > 0, "Train dataset is empty"
    assert dev_len > 0, "Dev dataset is empty"
    assert test_len > 0, "Test dataset is empty"
    total_len = train_len + dev_len + test_len
    assert total_len == jidenn_dataset.length, f"Total length is not correct: {total_len} != {jidenn_dataset.length}"
    assert train_len > dev_len and train_len > test_len, "Train dataset is not the largest"
