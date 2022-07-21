from jidenn_dataset import JIDENNDataset
from models import BasicFCModel, TransformerModel
import numpy as np
import tensorflow as tf
import argparse
import os
# import tensorflow_decision_forests as tfdf
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default


parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=6, type=int, help="Maximum number of threads to use.")
parser.add_argument("--debug", default=False, type=bool, help="Debug mode.")

#Dataset args
parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--take", default=None, type=int, help="Length of data to use.")
parser.add_argument("--shuffle", default=5000, type=int, help="Size of shuffling batch.")
parser.add_argument("--dev_size", default=0.2, type=float, help="Size of dev dataset.")

#BasicFCModel args
parser.add_argument("--hidden_layers", default=[16], nargs="*", type=int, help="Hidden layer sizes.")
parser.add_argument("--dropout", default=0.5, type=float, help="Dropout after FC layers.")

#TransformerModel args
parser.add_argument("--transformer_dropout", default=0., type=float, help="Transformer dropout.")
parser.add_argument("--transformer_expansion", default=4, type=float, help="Transformer FFN expansion factor.")
parser.add_argument("--transformer_heads", default=4, type=int, help="Transformer heads.")
parser.add_argument("--transformer_layers", default=2, type=int, help="Transformer layers.")
# parser.add_argument("--we_dim", default=64, type=int, help="Word embedding dimension.")

#RNN part args
parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
parser.add_argument("--rnn_cell_dim", default=128, type=int, help="RNN cell dimension.")


parser.add_argument("--label_smoothing", default=0., type=float, help="Smoothing of labels.")



def create_dataset(jidenn: JIDENNDataset,args:argparse.Namespace) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    def prep_dataset(data, label):
        label = tf.one_hot(tf.cast(label, tf.int32), len(JIDENNDataset.target_mapping.keys()))
        return data, label
    dataset = jidenn.train.dataset
    dataset = dataset.map(prep_dataset)
    size = dataset.cardinality()
    dataset = dataset.shuffle(args.shuffle if args.shuffle != 0 else size)
    dev = dataset.take(int(args.dev_size * tf.cast(size, tf.float64)))
    train = dataset.skip(int(args.dev_size * tf.cast(size, tf.float64)))
    train = train.batch(args.batch_size)
    dev = dev.batch(args.batch_size)
    train = train.prefetch(tf.data.AUTOTUNE)
    dev = dev.prefetch(tf.data.AUTOTUNE)
    return train, dev


def main(args: argparse.Namespace) -> None:
    if args.debug:
        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    filename = "data/user.pleskot.27852802.ANALYSIS._001685.root:NOMINAL"
    jidenn = JIDENNDataset([filename], load='tf_records/jidenn.tfrecord')


    train, dev = create_dataset(jidenn,args)


    model = BasicFCModel(args, len(jidenn.variables), jidenn.LABELS)

    model.summary()
    model.fit(train, validation_data=dev, epochs=args.epochs)
    
    print([np.argmax(prediction) for prediction in model.predict(dev.take(1))])


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
