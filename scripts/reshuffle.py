import os
import sys
sys.path.append(os.getcwd())
import argparse
import logging
from functools import partial
import tensorflow as tf
logging.basicConfig(format='[%(asctime)s][%(levelname)s] - %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
#

from jidenn.data.JIDENNDataset import JIDENNDataset


parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, help="Path to save the dataset")
parser.add_argument("--load_path", type=str, help="Path to the root file")
parser.add_argument("--shuffle", type=int, default=10_000_000, required=False, help="Shuffle buffer size")
parser.add_argument("--num_shards", type=int, default=256, required=False,
                    help="Number of shards to save the dataset in")


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.save_path, exist_ok=True)

    def reshuffle(load_path, save_path):
        dataset = JIDENNDataset.load(load_path)
        dataset = dataset.apply(lambda x: x.shuffle(args.shuffle).prefetch(tf.data.AUTOTUNE))
        dataset.save(save_path, num_shards=args.num_shards)

    # reshuffle 5 times to a temp file
    load_path = args.load_path
    for i in range(5):
        tmp_file = os.path.join(args.save_path, f'tmp', f'{i}')
        reshuffle(load_path, tmp_file)
        load_path = tmp_file

    # reshuffle to the final file
    reshuffle(load_path, args.save_path)
    logging.info(f'Dataset saved to {args.save_path}')

    os.rmdir(os.path.join(args.save_path, 'tmp'))
    dataset = JIDENNDataset.load(args.save_path)
    size = dataset.length
    dataset.plot_single_variable('jets_pt', os.path.join(args.save_path, 'pt.png'), bins=200,
                                 badge_text=f'N = {size:,}', badge=True, weight_variable='weight', multiple='stack',
                                 ylog=True, hue_variable='JZ_slice', xlabel=r'$p_\mathrm{T}$ [TeV]')


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
