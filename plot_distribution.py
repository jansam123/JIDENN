import tensorflow as tf
import os
import argparse
import json

from jidenn.data.JIDENNDataset import JIDENNDataset
from jidenn.const import LATEX_NAMING_CONVENTION

parser = argparse.ArgumentParser()
parser.add_argument("--load_path", type=str, help="Path to the saved tf.data.Dataset file")
parser.add_argument("--save_path", type=str, default='plots', help="Path to save the plots")
parser.add_argument("--take", type=int, default=10_000, help="Number of samples to plot")
parser.add_argument("-v", "--variables", type=str, nargs='*', help="Variables to plot")
parser.add_argument("--hue_variable", type=str, default='jets_PartonTruthLabelID',
                    help="Variable to use for hue. Needs to be categorical/integer.")

HUE_MAPPER = {1: 'quark', 2: 'quark', 3: 'quark', 4: 'quark', 5: 'quark', 6: 'quark', 21: 'gluon'}


def main(args: argparse.Namespace):
    dataset = JIDENNDataset.load(args.load_path)
    print(dataset.dataset.cardinality())
    dataset = dataset.apply(lambda x: x.take(args.take))
    os.makedirs(args.save_path, exist_ok=True)
    dataset.plot_data_distributions(folder=args.save_path,
                                    variables=args.variables + [args.hue_variable],
                                    hue_variable=args.hue_variable,
                                    named_labels=HUE_MAPPER if args.hue_variable == 'jets_PartonTruthLabelID' else None,
                                    xlabel_mapper=LATEX_NAMING_CONVENTION)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)