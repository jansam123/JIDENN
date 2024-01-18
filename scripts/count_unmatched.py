import os
import sys
sys.path.append(os.getcwd())
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(description='Count PFOs in a dataset')
parser.add_argument('--load_path', type=str, help='Path to dataset')


@tf.function
def count_unmatched(state, sample):
    reference_tensor = sample['jets_PartonTruthLabelID']
    values = tf.constant([-1, 1, 2, 3, 4, 5, 6, 21], dtype=tf.int32)
    mask = tf.math.equal(tf.expand_dims(reference_tensor, axis=1), tf.tile(
        tf.expand_dims(values, axis=0), [tf.shape(reference_tensor)[0], 1]))
    mask2 = tf.logical_and(tf.greater(sample['jets_pt'], 200_000), tf.less(sample['jets_pt'], 2_500_000))
    mask2 = tf.logical_and(mask2, tf.greater(sample['jets_eta'], -2.1))
    mask2 = tf.logical_and(mask2, tf.less(sample['jets_eta'], 2.1))
    mask = tf.reduce_any(mask, axis=1)
    mask = tf.logical_and(tf.logical_not(mask), mask2)
    n_unmatched = tf.reduce_sum(tf.cast(mask, tf.float32))
    n_pileup = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(reference_tensor, tf.constant(-1)), mask2), tf.float32))

    total_jets = tf.reduce_sum(tf.cast(mask2, tf.float32))
    return (state[0] + n_unmatched, state[1] + total_jets, state[2] + n_pileup)


def main(args):

    jz_total_jets, jz_big_jets, jz_pileup_jets = 0., 0., 0.

    print(30 * '-')
    for jz in [3, 4, 5, 6, 7]:
        path = os.path.join(args.load_path, f'JZ{jz}', 'dev')
        dataset = tf.data.Dataset.load(path, compression='GZIP')
        big_jets, total_jets, pileup_jets = dataset.reduce(
            (tf.constant(0.), tf.constant(0.), tf.constant(0.)), count_unmatched)

        print(f'JZ{jz}')
        print(f'Number of jets: {total_jets}')
        print(f'Number of jets with unmatched: {big_jets}')
        print(f'Fraction of jets with unmatched: {big_jets/total_jets:.3f}')
        print(f'Number of pileup jets: {pileup_jets}')
        print(f'Fraction of pileup jets: {pileup_jets/total_jets:.3f}')
        print(30 * '-')
        jz_total_jets += total_jets
        jz_big_jets += big_jets
        jz_pileup_jets += pileup_jets

    print(f'Number of jets: {jz_total_jets}')
    print(f'Number of jets with unmatched: {jz_big_jets}')
    print(f'Fraction of jets with unmatched: {jz_big_jets/jz_total_jets:.3f}')
    print(f'Number of pileup jets: {jz_pileup_jets}')
    print(f'Fraction of pileup jets: {jz_pileup_jets/jz_total_jets:.3f}')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
