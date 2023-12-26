from jidenn.data.JIDENNDataset import JIDENNDataset
import tensorflow as tf
import os
import argparse
import logging
logging.basicConfig(format='[%(asctime)s][%(levelname)s] - %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str,  help="Path to save the dataset")
parser.add_argument("--load_path", type=str,  help="Path to the root file")
parser.add_argument("--num_shards", type=int, default=256,
                    help="Number of shards to use when saving the dataset")
parser.add_argument("--train_frac", type=float, default=0.8,
                    help="Fraction of the dataset to use for training")
parser.add_argument("--dev_frac", type=float, default=0.1,
                    help="Fraction of the dataset to use for development")
parser.add_argument("--test_frac", type=float, default=0.1,
                    help="Fraction of the dataset to use for testing")


def main(args: argparse.Namespace) -> None:
    logging.info(
        f'Running with args: {{{", ".join([f"{k}: {v}" for k, v in vars(args).items()])}}}')
    os.makedirs(args.save_path, exist_ok=True)

    files = [os.path.join(args.load_path, file) for file in os.listdir(
        os.path.join(args.load_path)) if file.startswith('_') and len(os.listdir(os.path.join(args.load_path, file))) > 0]

    dataset = JIDENNDataset.load_multiple(files, mode='interleave')

    dataset = dataset.apply(lambda x: x.prefetch(
        tf.data.AUTOTUNE), preserves_length=True)
    dss = dataset.split_train_dev_test(
        args.train_frac, args.dev_frac, args.test_frac)

    for name, ds in zip(['test', 'dev', 'train'], dss):
        save_path = os.path.join(args.save_path, name)
        if os.path.exists(save_path):
            os.system(f'rm -rf {save_path}')
        os.makedirs(save_path, exist_ok=True)
        ds.save(save_path, num_shards=args.num_shards)
        logging.info(
            f'Saved {name} dataset to {save_path}')

    logging.info(f'DONE')


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
