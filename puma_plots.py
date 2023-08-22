from puma import VarVsEff, VarVsEffPlot
import pandas as pd
import os
import numpy as np
import argparse
from jidenn.config.eval_config import Binning

parser = argparse.ArgumentParser()
parser.add_argument("--load_path", type=str, help="Path to the saved tf.data.Dataset file")
parser.add_argument("--save_path", type=str, default='plots', help="Path to save the plots")
parser.add_argument("-v", "--variable", type=str, default='jets_pt', help="Variables to plot")
parser.add_argument("-w", "--weight", type=str, default=None, help="Weight variable")
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

HUE_MAPPER = {1: 'quark', 2: 'quark', 3: 'quark', 4: 'quark', 5: 'quark', 6: 'quark', 21: 'gluon'}


def main(args):
    score_dataset = pd.read_csv(args.load_path, index_col=0)
    is_quark = score_dataset['label'] == 1
    is_qluon = score_dataset['label'] == 0
    binning = Binning(min_bin=60_000, max_bin=2_500_000, bins=7, log_bin_base=0, variable=args.variable)

    if binning.log_bin_base is not None:
        min_val = np.log(binning.min_bin) / \
            np.log(binning.log_bin_base) if binning.log_bin_base != 0 else np.log(binning.min_bin)
        max_val = np.log(binning.max_bin) / \
            np.log(binning.log_bin_base) if binning.log_bin_base != 0 else np.log(binning.max_bin)
        bins = np.logspace(min_val, max_val,
                           binning.bins + 1, base=binning.log_bin_base if binning.log_bin_base != 0 else np.e)
    else:
        bins = np.linspace(binning.min_bin, binning.max_bin, binning.bins + 1)

    plot_bkg_rej = VarVsEffPlot(
        mode="bkg_rej",
        ylabel="Background rejection",
        xlabel=r"$p_{T}$ [GeV]",
        logy=False,
        logx=True if binning.log_bin_base is not None else False,
        # atlas_second_tag="$\\sqrt{s}=13$ TeV, dummy jets \ndummy sample, $f_{c}=0.018$",
        n_ratio_panels=1,
    )
    plot_sig_eff = VarVsEffPlot(
        mode="sig_eff",
        ylabel="Signal efficiency",
        xlabel=r"$p_{T}$ [GeV]",
        logy=False,
        logx=True if binning.log_bin_base is not None else False,
        # atlas_second_tag="$\\sqrt{s}=13$ TeV, dummy jets, \ndummy sample, $f_{c}=0.018$",
        n_ratio_panels=1,
    )

    plots = []
    for score_name in score_dataset.columns:
        if 'score' not in score_name:
            continue
        plot = VarVsEff(
            x_var_sig=score_dataset[args.variable][is_quark],
            disc_sig=score_dataset[score_name][is_quark],
            x_var_bkg=score_dataset[args.variable][is_qluon],
            disc_bkg=score_dataset[score_name][is_qluon],
            bins=bins,
            working_point=0.5,
            disc_cut=None,
            fixed_eff_bin=False,
            label=score_name.strip('_score'),
        )
        plot_bkg_rej.add(plot, reference=True if score_name == 'transformer_score' else False)
        plot_sig_eff.add(plot, reference=True if score_name == 'transformer_score' else False)

    plot_bkg_rej.draw()
    plot_sig_eff.draw()
    plot_bkg_rej.savefig(os.path.join(args.save_path, 'bkg_rej.png'), dpi=400)
    plot_sig_eff.savefig(os.path.join(args.save_path, 'sig_eff.png'), dpi=400)


if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
