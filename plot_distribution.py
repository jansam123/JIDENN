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
parser.add_argument("--save_path", type=str, default='plots', help="Path to save the plots")
parser.add_argument("--take", type=int, default=100_000, help="Number of samples to plot")
parser.add_argument("-v", "--variables", type=str, nargs='*', help="Variables to plot")
parser.add_argument("--hue_variable", type=str, default='jets_PartonTruthLabelID',
                    help="Variable to use for hue. Needs to be categorical/integer.")
parser.add_argument("--plot_single", type=bool, default=False, help="Whether to plot single variable distributions")
parser.add_argument("--bins", type=int, default=50, help="Number of bins for the histograms")
parser.add_argument("--log_bins", action='store_true', help="Whether to use log-spaced bins")

HUE_MAPPER = {1: 'quark', 2: 'quark', 3: 'quark', 4: 'quark', 5: 'quark', 6: 'quark', 21: 'gluon'}


def plot_single(dataset: tf.data.Dataset,
                variable: str,
                hue_var: str,
                log_bins: bool = False,
                ylog: bool = False,
                xlim: Optional[tuple] = (0.04, 4.),
                ylim: Optional[tuple] = None,
                bins: Optional[int] = None,
                hue_order: Optional[list] = None,
                save_path: Optional[str] = None,
                weight_var: Optional[str] = None,
                badge_text: Optional[str] = None):

    var_dataset = dataset.map(lambda x: x[variable])
    np_var = np.array(list(var_dataset.as_numpy_iterator()))
    hue_dataset = dataset.map(lambda x: x[hue_var])
    np_hue = np.array(list(hue_dataset.as_numpy_iterator()))
    df = pd.DataFrame({variable: np_var, hue_var: np_hue})

    if weight_var is not None:
        df['weight'] = np.array(list(dataset.map(lambda x: x[weight_var]).as_numpy_iterator()))
    df['Truth Label'] = df[hue_var].replace(HUE_MAPPER)

    print(df)
    if 'pt' in variable:
        df[variable] *= 1e-6

    if log_bins and isinstance(bins, int) and xlim is not None:
        bins = np.logspace(np.log(xlim[0]), np.log(xlim[1]), bins + 1, base=np.e)
    elif isinstance(bins, int) and xlim:
        bins = np.linspace(xlim[0], xlim[1], bins + 1)
    elif isinstance(bins, int) and not xlim:
        bins = bins
    else:
        bins = 'auto'

    # ylim = None
    plot_single_dist(df, variable=variable,
                     hue_var='Truth Label', hue_order=hue_order,
                     save_path=save_path, bins=bins,
                     ylog=ylog, xlog=log_bins, xlabel=r'$p_{\mathrm{T}}$ [TeV]',
                     badge_text=badge_text, ylim=ylim, weight_var='weight' if weight_var is not None else None
                     )


def main(args: argparse.Namespace):
    dataset = JIDENNDataset.load(args.load_path)
    ds_size = dataset.dataset.cardinality().numpy()
    print(f'Dataset size: {ds_size}')
    if args.take is not None and args.take > 0:
        dataset = dataset.apply(lambda x: x.take(args.take))
    # os.makedirs(args.save_path, exist_ok=True)
    if args.plot_single:
        plot_single(dataset.dataset,
                    variable=args.variables[0], hue_var=args.hue_variable,
                    save_path=args.save_path, badge_text=r'$N_{\mathrm{jets}}$ = ' +
                    f'{ds_size:,} \n' + r'$n_{\mathrm{bins}}$ = ' + f'{args.bins}',
                    log_bins=args.log_bins, bins=args.bins)
    else:
        dataset.plot_data_distributions(folder=args.save_path,
                                        variables=args.variables + [args.hue_variable],
                                        hue_variable=args.hue_variable,
                                        named_labels=HUE_MAPPER if args.hue_variable == 'jets_PartonTruthLabelID' else None,
                                        xlabel_mapper=LATEX_NAMING_CONVENTION)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
