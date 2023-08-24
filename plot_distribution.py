from typing import Optional
import tensorflow as tf
import os
import argparse
import json
import numpy as np
import pandas as pd

from jidenn.data.JIDENNDataset import JIDENNDataset
from jidenn.const import LATEX_NAMING_CONVENTION
from jidenn.evaluation.plotter import plot_single_dist

parser = argparse.ArgumentParser()
parser.add_argument("--load_path", type=str, help="Path to the saved tf.data.Dataset file")
parser.add_argument("--load_paths", type=str, nargs='*', help="Path to the saved tf.data.Dataset file")
parser.add_argument("--save_path", type=str, default='plots', help="Path to save the plots")
parser.add_argument("--take", type=int, default=100_000, help="Number of samples to plot")
parser.add_argument("-v", "--variables", type=str, nargs='*', help="Variables to plot")
parser.add_argument("-w", "--weight", type=str, default=None, help="Weight variable")
parser.add_argument("--hue_variable", type=str, default='jets_PartonTruthLabelID',
                    help="Variable to use for hue. Needs to be categorical/integer.")
parser.add_argument("--plot_single", action='store_true', help="Whether to plot a single distribution")
parser.add_argument("--bins", type=int, default=50, help="Number of bins for the histograms")
parser.add_argument("--log_bins", action='store_true', help="Whether to use log-spaced bins")
parser.add_argument("--ylog", action='store_true', help="Whether to use log-spaced bins")
parser.add_argument("--xlog", action='store_true', help="Whether to use log-spaced bins")
parser.add_argument("--no_badge", action='store_true', help="Whether to use log-spaced bins")
parser.add_argument("--xlim", type=float, nargs=2, default=None, help="X-axis limits")
parser.add_argument("--ylim", type=float, nargs=2, default=None, help="Y-axis limits")
parser.add_argument("--stat", type=str, default='count',
                    help="Statistic to plot. Options: count, density, probability.")
parser.add_argument("--multiple", type=str, default='layer',
                    help="How to plot multiple distributions. Options: layer, stack")
parser.add_argument("--shuffle", type=int, default=100_000, help="Shuffle buffer size")

HUE_MAPPER = {1: 'quark', 2: 'quark', 3: 'quark', 4: 'quark', 5: 'quark', 6: 'quark', 21: 'gluon'}


def plot_single(dataset: tf.data.Dataset,
                variable: str,
                hue_var: str,
                log_bins: bool = False,
                ylog: bool = False,
                xlog: bool = True,
                xlim: Optional[tuple] = None,
                ylim: Optional[tuple] = None,
                bins: Optional[int] = None,
                hue_order: Optional[list] = None,
                save_path: Optional[str] = None,
                weight_var: Optional[str] = None,
                stat: Optional[str] = 'count',
                multiple: str = 'layer',
                badge: bool = True,
                badge_text: Optional[str] = None):

    var_dataset = dataset.map(lambda x: x[variable])
    np_var = np.array(list(var_dataset.as_numpy_iterator()))
    hue_dataset = dataset.map(lambda x: x[hue_var])
    np_hue = np.array(list(hue_dataset.as_numpy_iterator()))
    df = pd.DataFrame({variable: np_var, hue_var: np_hue})

    if weight_var is not None:
        df['weight'] = np.array(list(dataset.map(lambda x: x[weight_var]).as_numpy_iterator()))
    df[hue_var] = df[hue_var].replace(HUE_MAPPER) if hue_var == 'jets_PartonTruthLabelID' else df[hue_var]

    print(df)
    if 'pt' in variable:
        df[variable] *= 1e-6

    plot_single_dist(df, variable=variable, xlim=xlim, log_bins=log_bins,
                     hue_var=hue_var, hue_order=hue_order, badge=badge,
                     save_path=save_path, bins=bins, stat=stat, multiple=multiple,
                     ylog=ylog, xlog=xlog, xlabel=r'$p_{\mathrm{T}}$ [TeV]',
                     badge_text=badge_text, ylim=ylim, weight_var='weight' if weight_var is not None else None
                     )


def main(args: argparse.Namespace):
    if args.load_paths is None and args.load_path is not None:
        print(f'File: {args.load_path}')
        dataset = JIDENNDataset.load(args.load_path)
        if args.take is not None and args.take > 0:
            dataset = dataset.take(args.take)
        dataset = dataset.apply(lambda x: x.shuffle(args.shuffle).prefetch(tf.data.AUTOTUNE))
    elif args.load_paths is not None and args.load_path is None:
        print(f'Files: {args.load_paths}')
        dataset = JIDENNDataset.load_parallel(args.load_paths, take=None if args.take == 0 else args.take)
        dataset = dataset.apply(lambda x: x.shuffle(args.shuffle).prefetch(tf.data.AUTOTUNE))
    else:
        raise ValueError('Either load_path or load_paths needs to be specified.')

    ds = dataset.dataset
    ds_size = ds.cardinality().numpy()
    print(f'Dataset size: {ds_size:,}')

    # badge_text = r'$N_{\mathrm{jets}}$ = ' + f'{ds_size:,} \n' if ds_size is not None else None

    if args.plot_single:
        plot_single(ds,
                    variable=args.variables[0], hue_var=args.hue_variable,
                    save_path=args.save_path,
                    # badge_text='$N_{\mathrm{jets}}$ = ' + f'{ds_size:,} \n',
                    multiple=args.multiple,
                    badge=not args.no_badge,
                    weight_var=args.weight, ylog=args.ylog, ylim=args.ylim, xlim=args.xlim, stat=args.stat,
                    xlog=args.xlog, log_bins=args.log_bins, bins=args.bins)
    else:
        dataset.plot_data_distributions(folder=args.save_path,
                                        variables=args.variables + [args.hue_variable],
                                        hue_variable=args.hue_variable,
                                        named_labels=HUE_MAPPER if args.hue_variable == 'jets_PartonTruthLabelID' else None,
                                        xlabel_mapper=LATEX_NAMING_CONVENTION)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
