import argparse
import os
from jidenn_dataset import JIDENNDataset
import tensorflow as tf
import time
import itertools

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--data", default='data', type=str, help="Path to data folder containing folder of .root files.")


def test(args: argparse.Namespace, datafiles:list[str]):
    
    jidenn = JIDENNDataset(datafiles, reading_size=args.reading_size, num_workers=args.num_workers, dev_size=0.)
    def prep_dataset(data, label):
        label = tf.one_hot(tf.cast(label, tf.int32), JIDENNDataset.LABELS)
        return data, label
    dataset = jidenn.train.dataset
    dataset = dataset.map(prep_dataset)
    dataset = dataset.take(args.take) if args.take is not None else dataset
    
    start = time.time()
    size = 0
    print('Size:')
    for _ in dataset:
        size += 1
        print(f'{size:_}', end='\r')
    end = time.time()
    size_reading_time = end - start
    print(f'size={size:_}, size_time={size_reading_time:.3f}s')
    dataset = dataset.batch(args.batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    start = time.time()
    a = 0
    print('Data:')
    for dato in dataset: 
        a += 1
        print(dato, end='\r')
        print(end='\r')
        print(a, end='\r')
    end = time.time()
    dataset_time = end - start
    print(f'batched_size={a:_}, size_time={dataset_time:.3f}s')
    return size_reading_time, dataset_time, size


def main(args: argparse.Namespace) -> None:
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    
    datafiles = [os.path.join(args.data, folder, file) for folder in os.listdir(args.data) for file in os.listdir(os.path.join(args.data, folder)) if '.root' in file]
    
    batch_sizes = [4096]
    reading_sizes = [1_000]
    num_workers = [4]
    filename = 'test_dataset.txt'
    
    for take in [None]:
        args.take = take
        for batch_size, reading_size, num_worker in itertools.product(batch_sizes, reading_sizes, num_workers):
            args.batch_size = batch_size
            args.reading_size = reading_size
            args.num_workers = num_worker

            size_reading_time, dataset_time, size = test(args, datafiles)
            
            with open(filename, 'a') as f:
                print(f'Batch size: {batch_size}, reading size: {reading_size}, num workers: {num_worker}, take: {take}, size:{size}', file=f)
                print(f'size_time={size_reading_time:.3f}s, loop_time={dataset_time:.3f}s', file=f)
                print(file=f)
            
    
    
    
if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
        
    