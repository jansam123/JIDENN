import os
import sys
sys.path.append(os.getcwd())
import tensorflow as tf
import argparse
import logging
logging.basicConfig(format='[%(asctime)s][%(levelname)s] - %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

from jidenn.data.JIDENNDataset import JIDENNDataset

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
parser.add_argument("--start_identifier", type=str, default='_',
                    help="Identifier for the start of the files to combine")
parser.add_argument("--mode", type=str, default='interleave',
                    help="Mode to use when combining the files")
parser.add_argument("--add_lund_plane", action='store_true',
                    help="Add Lund plane to the dataset")

def main(args: argparse.Namespace) -> None:
    logging.info(
        f'Running with args: {{{", ".join([f"{k}: {v}" for k, v in vars(args).items()])}}}')
    os.makedirs(args.save_path, exist_ok=True)

    files = [os.path.join(args.load_path, file) for file in os.listdir(
        os.path.join(args.load_path)) if file.startswith(args.start_identifier) and len(os.listdir(os.path.join(args.load_path, file))) > 0]
    if len(files) == 0:
        logging.error(
            f'No files found in {args.load_path}')
        raise ValueError(
            f'No files found in {args.load_path}')

    dataset = JIDENNDataset.load_multiple(files, mode=args.mode)
    
    if args.add_lund_plane:
        from jidenn.data.jet_reclustering import tf_create_lund_plane
        @tf.function
        def add_lundplane(x):
            out = x.copy()
            lund_graph, node_coords = tf.py_function(
                func=tf_create_lund_plane,
                inp=[x['part_px'], x['part_py'], x['part_pz'], x['part_energy']],
                Tout=[tf.TensorSpec(shape=(None, None), dtype=tf.float32), tf.TensorSpec(shape=(None, 4), dtype=tf.float32)]
            )
            out['lund_graph'] = lund_graph
            out['lund_graph_node_px'] = node_coords[:,0]
            out['lund_graph_node_py'] = node_coords[:,1]
            out['lund_graph_node_pz'] = node_coords[:,2]
            out['lund_graph_node_energy'] = node_coords[:,3]
            return out
        dataset = dataset.remap_data(add_lundplane)

    dataset = dataset.apply(lambda x: x.prefetch(
        tf.data.AUTOTUNE), preserves_length=True)
    
    if args.train_frac == 1.:
        dataset.save(args.save_path, num_shards=args.num_shards)
        logging.info(
            f'Saved whole dataset to {args.save_path}')
        return
    
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




