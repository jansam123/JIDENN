import pytest
import tempfile
import os
import numpy as np
import tensorflow as tf
import logging
import pandas as pd
logging.basicConfig(level=logging.INFO)

from jidenn.data.JIDENNDataset import JIDENNDataset, dict_to_stacked_tensor, FLOAT_PRECISION


def test_dict_to_stacked_tensor():
    # Test converting a dictionary to a stacked tensor
    data = {
        'feature1': tf.random.uniform((100,)),
        'feature2': tf.random.uniform((100,)),
        'feature3': tf.random.uniform((100,), maxval=10, dtype=tf.int32),
    }
    stacked_tensor = dict_to_stacked_tensor(data)
    assert isinstance(stacked_tensor, tf.Tensor), "Stacked tensor is not a tensor"
    assert stacked_tensor.shape == (100, 3), "Shape of stacked tensor is not correct"
    assert tf.reduce_all(stacked_tensor[:, 0] == data['feature1']), "Feature1 is not the same"
    assert tf.reduce_all(stacked_tensor[:, 1] == data['feature2']), "Feature2 is not the same"
    assert tf.reduce_all(stacked_tensor[:, 2] == tf.cast(data['feature3'], FLOAT_PRECISION)), "Feature3 is not the same"


def test_dict_to_stacked_tensor_interaction():
    data = {
        'feature1': np.random.rand(10),
        'feature2': np.random.rand(10),
        'feature3': np.random.randint(10, size=10),
    }
    data2 = {
        'feature1': np.random.rand(10, 10),
        'feature2': np.random.rand(10, 10),
        'feature3': np.random.randint(10, size=(10, 10)),
    }
    stacked_tensor = dict_to_stacked_tensor((data, data2))

    assert isinstance(stacked_tensor, tuple), "Stacked tensor is not a tensor"
    assert isinstance(stacked_tensor[0], tf.Tensor), "Stacked tensor is not a tensor"
    assert isinstance(stacked_tensor[1], tf.Tensor), "Interaction Stacked tensor is not a tensor"

    assert stacked_tensor[0].shape == (10, 3), "Shape of stacked tensor is not correct"
    assert np.all(stacked_tensor[0][:, 0] == data['feature1']), "Feature1 is not the same"
    assert np.all(stacked_tensor[0][:, 1] == data['feature2']), "Feature2 is not the same"
    assert np.all(stacked_tensor[0][:, 2] == data['feature3']), "Feature3 is not the same"

    assert stacked_tensor[1].shape == (10, 10, 3), "Interaction Shape of stacked tensor is not correct"
    assert np.all(stacked_tensor[1][:, :, 0] == data2['feature1']), "Interaction Feature1 is not the same"
    assert np.all(stacked_tensor[1][:, :, 1] == data2['feature2']), "Interaction Feature2 is not the same"
    assert np.all(stacked_tensor[1][:, :, 2] == data2['feature3']), "Interaction Feature3 is not the same"


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
