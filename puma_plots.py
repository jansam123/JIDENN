from puma import VarVsEff, VarVsEffPlot, RocPlot, Roc
from puma.metrics import calc_rej
import pandas as pd
import os
import numpy as np
import seaborn as sns
import argparse

from jidenn.config.eval_config import Binning
from jidenn.const import MODEL_NAMING_SCHEMA

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
    score_dataset[args.variable] = score_dataset[args.variable] * \
        1e-3 if args.variable == 'jets_pt' else score_dataset[args.variable]
    is_quark = score_dataset['label'] == 1
    is_gluon = score_dataset['label'] == 0
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
        ylabel=r"$\varepsilon_g^{-1}$",
        xlabel=r"$p_{T}$ [GeV]",
        logy=False,
        leg_ncol=2,
        # logx=True if binning.log_bin_base is not None else False,
        atlas_second_tag="50% WP",
        figsize=(7, 5),
        draw_errors=False,
        n_ratio_panels=0,
    )
    plot_sig_eff = VarVsEffPlot(
        mode="sig_rej",
        ylabel="Gluon efficiency",
        xlabel=r"$p_{T}$ [GeV]",
        leg_ncol=2,
        logy=False,
        # logx=True if binning.log_bin_base is not None else False,
        atlas_second_tag="50% WP",
        figsize=(7, 5),
        draw_errors=False,
        n_ratio_panels=0,
    )
    plot_bkg_rej_80 = VarVsEffPlot(
        mode="bkg_rej",
        ylabel=r"$\varepsilon_g^{-1}$",
        xlabel=r"$p_{T}$ [GeV]",
        logy=False,
        leg_ncol=2,
        # logx=True if binning.log_bin_base is not None else False,
        atlas_second_tag="80% WP",
        figsize=(7, 5),
        draw_errors=False,
        n_ratio_panels=0,
    )
    plot_sig_eff_80 = VarVsEffPlot(
        mode="sig_rej",
        ylabel="Gluon efficiency",
        xlabel=r"$p_{T}$ [GeV]",
        leg_ncol=2,
        logy=False,
        # logx=True if binning.log_bin_base is not None else False,
        atlas_second_tag="80% WP",
        figsize=(7, 5),
        draw_errors=False,
        n_ratio_panels=0,
    )

    plot_roc = RocPlot(
        n_ratio_panels=0,
        ylabel=r"$\varepsilon_g^{-1}$",
        xlabel=r"$\varepsilon_q$",
        atlas_second_tag="13 TeV",
        figsize=(5, 4),
        ymin=1,
        ymax=2e3,
        y_scale=1.4,
        grid=False,
        label_fontsize=14,
        fontsize=12,
        atlas_fontsize=11,
        leg_fontsize=11,
    )

    plots = []
    sig_eff = np.linspace(0.1, 1, 100)
    # score_dataset.columns:
    scores =["idepart_score", "ipart_score", "particle_net_score", "depart_score", "transformer_score", "part_score", "pfn_score", "highway_score","fc_score", "efn_score",]
    colours = sns.color_palette('coolwarm', len(scores))
    for i, score_name in enumerate(scores):
        if 'score' not in score_name:
            continue
        label = MODEL_NAMING_SCHEMA[score_name.replace('_score', '')]
        plot = VarVsEff(
            x_var_sig=score_dataset[args.variable][is_quark],
            disc_sig=score_dataset[score_name][is_quark],
            x_var_bkg=score_dataset[args.variable][is_gluon],
            disc_bkg=score_dataset[score_name][is_gluon],
            # bins=bins,
            bins=[200, 300, 400, 600, 850, 1100, 1400, 1750, 2500],
            working_point=0.5,
            disc_cut=None,
            fixed_eff_bin=False,
            marker='o',
            markersize=4,
            is_marker=True,
            label=label,

        )
        plot_80 = VarVsEff(
            x_var_sig=score_dataset[args.variable][is_quark],
            disc_sig=score_dataset[score_name][is_quark],
            x_var_bkg=score_dataset[args.variable][is_gluon],
            disc_bkg=score_dataset[score_name][is_gluon],
            # bins=bins,
            bins=[200, 300, 400, 600, 850, 1100, 1400, 1750, 2500],
            working_point=0.8,
            disc_cut=None,
            fixed_eff_bin=False,
            marker='o',
            markersize=4,
            is_marker=True,
            label=label,

        )
        rejs = calc_rej(score_dataset[score_name][is_quark], score_dataset[score_name][is_gluon], sig_eff)
        plot_roc.add_roc(
            Roc(
                sig_eff,
                rejs,
                n_test=sum(is_gluon),

                # rej_class="gluon",
                # signal_class="quark",
                label=label,
                colour=colours[i],
            ),
            reference=True if score_name == 'transformer_score' else False
        )
        plot_bkg_rej.add(plot, reference=True if score_name == 'transformer_score' else False)
        plot_bkg_rej_80.add(plot_80, reference=True if score_name == 'transformer_score' else False)
        plot_sig_eff.add(plot, reference=True if score_name == 'transformer_score' else False)
        plot_sig_eff_80.add(plot_80, reference=True if score_name == 'transformer_score' else False)

    plot_bkg_rej.draw()
    plot_bkg_rej_80.draw()
    plot_sig_eff.draw()
    plot_sig_eff_80.draw()
    plot_roc.draw()
    plot_roc.savefig(os.path.join(args.save_path, 'roc.png'), transparent=False, dpi=400)
    plot_bkg_rej.savefig(os.path.join(args.save_path, 'bkg_rej_50.png'), dpi=400)
    plot_bkg_rej_80.savefig(os.path.join(args.save_path, 'bkg_rej_80.png'), dpi=400)
    plot_sig_eff.savefig(os.path.join(args.save_path, 'sig_eff_50.png'), dpi=400)
    plot_sig_eff_80.savefig(os.path.join(args.save_path, 'sig_eff_80.png'), dpi=400)
    plot_roc.savefig(os.path.join(args.save_path, 'roc.pdf'), transparent=False, dpi=400)
    plot_bkg_rej.savefig(os.path.join(args.save_path, 'bkg_rej_50.pdf'), dpi=400)
    plot_bkg_rej_80.savefig(os.path.join(args.save_path, 'bkg_rej_80.pdf'), dpi=400)
    plot_sig_eff.savefig(os.path.join(args.save_path, 'sig_eff_50.pdf'), dpi=400)
    plot_sig_eff_80.savefig(os.path.join(args.save_path, 'sig_eff_80.pdf'), dpi=400)


if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
