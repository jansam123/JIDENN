import tensorflow as tf
import os
import argparse
import pickle
import logging

parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, help="Path to save the dataset")
parser.add_argument("--load_path", type=str, help="Path to the root file")
parser.add_argument("--num_shards", type=int, help="Path to the root file")


def main(args: argparse.Namespace) -> None:
    TEST_SIZE = 0.1
    DEV_SIZE = 0.1
    TRAIN_SIZE = 1 - TEST_SIZE - DEV_SIZE

    with open(os.path.join(args.load_path, 'element_spec'), 'rb') as f:
        element_spec = pickle.load(f)

    DATASET_SIZE = args.num_shards
    train_size = int(TRAIN_SIZE * DATASET_SIZE)
    val_size = int(DEV_SIZE * DATASET_SIZE)
    test_size = int(TEST_SIZE * DATASET_SIZE)

    @tf.function
    def is_test(x, y):
        return x % 5 == 0

    @tf.function
    def is_train(x, y):
        return not is_test(x, y)

    @tf.function
    def recover(x, y): 
        return y

    @tf.function
    def is_dev(x, y):
        return x % 2 == 0

    @tf.function
    def is_not_dev(x, y):
        return not is_dev(x, y)
    
    @tf.function
    def train_reader_func(full_dataset: tf.data.Dataset) -> tf.data.Dataset:
        train_dataset = full_dataset.enumerate() \
            .filter(is_train) \
            .map(recover)
        return train_dataset.interleave(lambda x: x, num_parallel_calls=tf.data.AUTOTUNE)

    @tf.function
    def dev_reader_func(full_dataset: tf.data.Dataset) -> tf.data.Dataset:
        
        dev_test_dataset = full_dataset.enumerate() \
            .filter(is_test) \
            .map(recover)

        dev_dataset = dev_test_dataset.enumerate() \
            .filter(is_dev) \
            .map(recover)        
        return dev_dataset.interleave(lambda x: x, num_parallel_calls=tf.data.AUTOTUNE)

    @tf.function
    def test_reader_func(full_dataset: tf.data.Dataset) -> tf.data.Dataset:
        dev_test_dataset = full_dataset.enumerate() \
            .filter(is_test) \
            .map(recover)
        test_dataset = dev_test_dataset.enumerate() \
            .filter(is_not_dev) \
            .map(recover)
        return test_dataset.interleave(lambda x: x, num_parallel_calls=tf.data.AUTOTUNE)


    for name, size, func in zip(['test', 'dev', 'train'], [TEST_SIZE, DEV_SIZE, TRAIN_SIZE], [test_reader_func, dev_reader_func, train_reader_func]):
        os.makedirs(os.path.join(args.save_path, name), exist_ok=True)

        @tf.function
        def random_shards(_) -> tf.Tensor:
            return tf.random.uniform(shape=[], minval=0, maxval=int(size * args.num_shards), dtype=tf.int64)

        ds = tf.data.experimental.load(args.load_path, compression='GZIP', element_spec=element_spec, reader_func=func)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        
        tf.data.experimental.save(ds, os.path.join(args.save_path, name),
                                  compression='GZIP', shard_func=random_shards)
        
        # loaded_ds = tf.data.experimental.load(os.path.join(args.save_path, name),
        #                                       compression='GZIP', element_spec=element_spec)
        # print(loaded_ds.cardinality().numpy())
        # for i in loaded_ds.take(1):
        #     print(i)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
