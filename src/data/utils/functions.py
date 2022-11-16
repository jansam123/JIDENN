import tensorflow as tf



# def split_by_size(self, size: float) -> tuple[ROOTDataset, ROOTDataset]:
#     return ROOTDataset(self._dataset.take(int(size * self._dataset.cardinality().numpy())), self._variables), ROOTDataset(self._dataset.skip(int(size * self._dataset.cardinality().numpy())), self._variables)

def split_dataset(dataset: tf.data.Dataset, size:float) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    return dataset.take(int(size * dataset.cardinality().numpy())), dataset.skip(int(size * dataset.cardinality().numpy()))
    

def split_train_dev_test(self, test_size: float, dev_size: float) -> tuple[ROOTDataset, ROOTDataset, ROOTDataset]:
    train_size = 1 - test_size - dev_size
    train, dev_test = self.split_by_size(train_size)
    dev, test = dev_test.split_by_size(dev_size / (1 - train_size))
    return train, dev, test
