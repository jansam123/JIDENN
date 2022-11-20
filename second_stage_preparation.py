#!/home/jankovys/JIDENN/venv/bin/python
import tensorflow as tf
import os
import argparse
import pickle
import logging
#
logging.basicConfig(level=logging.INFO)
# from src.data.ROOTDataset import ROOTDataset, ROOTVariables
ROOTVariables = dict[str, tf.RaggedTensor]

parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, help="Path to save the dataset")
parser.add_argument("--file_path", type=str, help="Path to the root file")
parser.add_argument("--num_shards", type=int, default=96, help="Path to the root file")

tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)


@tf.function
def no_pile_up(sample: ROOTVariables) -> ROOTVariables:
    sample = sample.copy()
    jet_mask = tf.math.greater(sample['jets_PartonTruthLabelID'], 0)
    for key, item in sample.items():
        if key.startswith('jets_') and key != 'jets_n':
            sample[key] = tf.ragged.boolean_mask(item, jet_mask)
        sample[key] = tf.cast(sample[key], tf.int32 if 'int' in item.dtype.name else tf.float32)
    return sample


@tf.function
def filter_empty_jets(sample: ROOTVariables) -> tf.Tensor:
    return tf.greater(tf.size(sample['jets_PartonTruthLabelID']), 0)


@tf.function
def flatten_toJet(sample: ROOTVariables) -> ROOTVariables:
    sample = sample.copy()
    jet_shape = tf.shape(sample['jets_PartonTruthLabelID'])
    for key, item in sample.items():
        if isinstance(item, tf.RaggedTensor):
            continue
        elif len(tf.shape(item)) == 0:
            sample[key] = tf.tile(item[tf.newaxis, tf.newaxis], [jet_shape[0], 1])
        elif tf.shape(item)[0] != jet_shape[0]:
            sample[key] = tf.tile(item[tf.newaxis, :], [jet_shape[0], 1])
        else:
            continue
    return tf.data.Dataset.from_tensor_slices(sample)


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.save_path, exist_ok=True)
    base_dir = args.file_path
    files = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.startswith('_')]
    logging.info(f"Found {len(files)} files")
    logging.info(f"Files: {files}")

    def load_dataset_file(element_spec, file: str) -> tf.data.Dataset:
        root_dt = tf.data.experimental.load(file, compression='GZIP', element_spec=element_spec)
        root_dt = root_dt.map(no_pile_up, num_parallel_calls=tf.data.AUTOTUNE,
                              deterministic=False).filter(filter_empty_jets)
        root_dt = root_dt.prefetch(tf.data.AUTOTUNE)
        return root_dt
    
    with open(os.path.join(files[0], 'element_spec'), 'rb') as f:
        element_spec = pickle.load(f)
    dataset = load_dataset_file(element_spec, files[0])
    
    for file in files[1:]:
        with open(os.path.join(file, 'element_spec'), 'rb') as f:
            element_spec = pickle.load(f)
        dataset = dataset.concatenate(load_dataset_file(element_spec, file))
        
    dataset = dataset.interleave(flatten_toJet, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    dataset = dataset.shuffle(1000, reshuffle_each_iteration=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    with open(os.path.join(args.save_path, 'element_spec'), 'wb') as f:
        pickle.dump(dataset.element_spec, f)

    @tf.function
    def random_shards(_: ROOTVariables) -> tf.Tensor:
        return tf.random.uniform(shape=[], minval=0, maxval=args.num_shards, dtype=tf.int64)

    tf.data.experimental.save(dataset, path=args.save_path, compression='GZIP', shard_func=random_shards)
    dataset = tf.data.experimental.load(args.save_path, compression='GZIP')
    for i in dataset.take(1):
        print(i)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
