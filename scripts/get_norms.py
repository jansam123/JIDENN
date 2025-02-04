import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, required=True)

def main(args):
    print(args.path)
    dataset = tf.data.Dataset.load(args.path, compression='GZIP')
    norm = dataset.reduce(0., lambda x, y: x + y['weight_flat'])
    print(norm)
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
