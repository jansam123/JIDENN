
import os
import sys
sys.path.append(os.getcwd())
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(description='Count PFOs in a dataset')
parser.add_argument('--load_path', type=str, help='Path to dataset')



@tf.function
def count_PFOs(state, sample):
    n_PFOs = tf.reduce_sum(tf.ones_like(sample['jets_PFO_pt']))
    weight = sample['weight_spectrum']
    if n_PFOs > 100:
        return state[0] + weight, state[1] + weight
    else:
        return state[0], state[1] + weight
    


def main(args):
    dataset = tf.data.Dataset.load(args.load_path, compression='GZIP')
    big_jets, total_jets = dataset.reduce((tf.constant(0.0), tf.constant(0.0)), count_PFOs)
    
    print(f'Number of jets: {total_jets}')
    print(f'Number of jets with PFOs: {big_jets}')
    print(f'Fraction of jets with PFOs: {100*big_jets/total_jets}%')
    

    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
    
    
    