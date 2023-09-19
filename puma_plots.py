from puma import VarVsEff, VarVsEffPlot, RocPlot, Roc
from puma.metrics import calc_rej
import pandas as pd
import os
import numpy as np
import seaborn as sns
import argparse
from sklearn.metrics import roc_curve, precision_recall_curve, precision_score

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
parser.add_argument("--cut", type=str, default=None,
                    help="Cut to apply to the dataset.")
parser.add_argument("--title", type=str, default=None,
                    help="Cut to apply to the dataset.")
parser.add_argument("--pt_bins", type=int, default=None,
                    nargs='+', help="Bins for the pt plots")

HUE_MAPPER = {1: 'quark', 2: 'quark', 3: 'quark', 4: 'quark', 5: 'quark', 6: 'quark', 21: 'gluon'}


def main(args):
    os.makedirs(args.save_path, exist_ok=True)
    score_dataset = pd.read_csv(args.load_path, index_col=0)
    score_dataset = score_dataset.query(args.cut) if args.cut is not None else score_dataset
    score_dataset['jets_pt'] = score_dataset['jets_pt'] * \
        1e-3 if 'jets_pt' == 'jets_pt' else score_dataset['jets_pt']
    is_quark = score_dataset['label'] == 1
    is_gluon = score_dataset['label'] == 0

    plot_bkg_rej = VarVsEffPlot(
        mode="bkg_rej",
        ylabel=r"$\varepsilon_g^{-1}$",
        xlabel=r"$p_{T}$ [GeV]",
        logy=False,
        leg_ncol=2,
        xmin=200,
        xmax=2500,
        ymin=6,
        ymax=35,
        label_fontsize=14,
        fontsize=12,
        leg_fontsize=11,
        # logx=True if binning.log_bin_base is not None else False,
        atlas_second_tag="13 TeV, 50% WP" if args.title is None else f"13 TeV, 50% WP, {args.title}",
        figsize=(7, 5),
        draw_errors=False,
        n_ratio_panels=0,
    )
    plot_bkg_rej_eta = VarVsEffPlot(
        mode="bkg_rej",
        ylabel=r"$\varepsilon_g^{-1}$",
        xlabel=r"$\|\eta\|$",
        leg_ncol=2,
        logy=False,
        label_fontsize=14,
        fontsize=12,
        leg_fontsize=11,
        # logx=True if binning.log_bin_base is not None else False,
        atlas_second_tag="13 TeV, 50% WP" if args.title is None else f"13 TeV, 50% WP, {args.title}",
        figsize=(7, 5),
        draw_errors=False,
        n_ratio_panels=0,
    )
    plot_bkg_rej_mu = VarVsEffPlot(
        mode="bkg_rej",
        ylabel=r"$\varepsilon_g^{-1}$",
        xlabel=r"$\mu$",
        leg_ncol=2,
        logy=False,
        label_fontsize=14,
        fontsize=12,
        leg_fontsize=11,
        # logx=True if binning.log_bin_base is not None else False,
        atlas_second_tag="13 TeV, 50% WP" if args.title is None else f"13 TeV, 50% WP, {args.title}",
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
        xmin=200,
        xmax=2500,
        ymin=2,
        ymax=6.2,
        label_fontsize=14,
        fontsize=12,
        leg_fontsize=11,
        # logx=True if binning.log_bin_base is not None else False,
        atlas_second_tag="13 TeV, 80% WP" if args.title is None else f"13 TeV, 80% WP, {args.title}",
        figsize=(7, 5),
        draw_errors=False,
        n_ratio_panels=0,
    )
    plot_bkg_rej_80_eta = VarVsEffPlot(
        mode="bkg_rej",
        ylabel=r"$\varepsilon_g^{-1}$",
        xlabel=r"$\|\eta\|$",
        leg_ncol=2,
        logy=False,
        # logx=True if binning.log_bin_base is not None else False,
        atlas_second_tag="13 TeV, 80% WP" if args.title is None else f"13 TeV, 80% WP, {args.title}",
        figsize=(7, 5),
        label_fontsize=14,
        fontsize=12,
        leg_fontsize=11,
        draw_errors=False,
        n_ratio_panels=0,
    )
    plot_bkg_rej_80_mu = VarVsEffPlot(
        mode="bkg_rej",
        ylabel=r"$\varepsilon_g^{-1}$",
        xlabel=r"$\mu$",
        leg_ncol=2,
        logy=False,
        label_fontsize=14,
        fontsize=12,
        leg_fontsize=11,
        # logx=True if binning.log_bin_base is not None else False,
        atlas_second_tag="13 TeV, 80% WP" if args.title is None else f"13 TeV, 80% WP, {args.title}",
        figsize=(7, 5),
        draw_errors=False,
        n_ratio_panels=0,
    )

    plot_roc = RocPlot(
        n_ratio_panels=0,
        ylabel=r"$\varepsilon_g^{-1}$",
        xlabel=r"$\varepsilon_q$",
        atlas_second_tag="13 TeV" if args.title is None else f"13 TeV, {args.title}",
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
    # scores = ["idepart_score", "ipart_score", "particle_net_score", "depart_score",
    #           "transformer_score", "part_score", "pfn_score", "highway_score", "fc_score", "efn_score",]
    scores = ["idepart_score", "ipart_score", "particle_net_score", "pfn_score", "highway_score", "fc_score", "efn_score",]
    score_dataset['jets_eta'] = score_dataset['jets_eta'].abs()
    colors = sns.color_palette('coolwarm', len(scores))
    colours = sns.color_palette('coolwarm', len(scores))
    for i, score_name in enumerate(scores):
        if 'score' not in score_name:
            continue
        label = MODEL_NAMING_SCHEMA[score_name.replace('_score', '')]
        plot = VarVsEff(
            x_var_sig=score_dataset['jets_pt'][is_quark],
            disc_sig=score_dataset[score_name][is_quark],
            weights_sig=score_dataset[args.weight][is_quark] if args.weight is not None else None,
            x_var_bkg=score_dataset['jets_pt'][is_gluon],
            disc_bkg=score_dataset[score_name][is_gluon],
            weights_bkg=score_dataset[args.weight][is_gluon] if args.weight is not None else None,
            # bins=bins,
            bins=args.pt_bins if args.pt_bins is not None else [200, 300, 400, 600, 850, 1100, 1400, 1750, 2500],
            working_point=0.5,
            disc_cut=None,
            fixed_eff_bin=True,
            marker='o',
            markersize=4,
            is_marker=True,
            label=label,
            colour=colours[i],

        )
        plot_80 = VarVsEff(
            x_var_sig=score_dataset['jets_pt'][is_quark],
            disc_sig=score_dataset[score_name][is_quark],
            weights_sig=score_dataset[args.weight][is_quark] if args.weight is not None else None,
            x_var_bkg=score_dataset['jets_pt'][is_gluon],
            disc_bkg=score_dataset[score_name][is_gluon],
            weights_bkg=score_dataset[args.weight][is_gluon] if args.weight is not None else None,
            # bins=bins,
            bins=args.pt_bins if args.pt_bins is not None else [200, 300, 400, 600, 850, 1100, 1400, 1750, 2500],
            working_point=0.8,
            disc_cut=None,
            fixed_eff_bin=True,
            marker='o',
            markersize=4,
            is_marker=True,
            label=label,
            colour=colours[i],

        )
        plot_eta = VarVsEff(
            x_var_sig=score_dataset['jets_eta'][is_quark],
            disc_sig=score_dataset[score_name][is_quark],
            weights_sig=score_dataset[args.weight][is_quark] if args.weight is not None else None,
            x_var_bkg=score_dataset['jets_eta'][is_gluon],
            disc_bkg=score_dataset[score_name][is_gluon],
            weights_bkg=score_dataset[args.weight][is_gluon] if args.weight is not None else None,
            # bins=bins,
            bins=[0., 0.1, 0.2, 0.4, 0.6, 0.8, 1., 1.4, 2.1],
            working_point=0.5,
            disc_cut=None,
            fixed_eff_bin=True,
            marker='o',
            markersize=4,
            is_marker=True,
            label=label,
            colour=colours[i],

        )
        plot_80_eta = VarVsEff(
            x_var_sig=score_dataset['jets_eta'][is_quark],
            disc_sig=score_dataset[score_name][is_quark],
            weights_sig=score_dataset[args.weight][is_quark] if args.weight is not None else None,
            x_var_bkg=score_dataset['jets_eta'][is_gluon],
            disc_bkg=score_dataset[score_name][is_gluon],
            weights_bkg=score_dataset[args.weight][is_gluon] if args.weight is not None else None,
            # bins=bins,
            bins=[0., 0.1, 0.2, 0.4, 0.6, 0.8, 1., 1.4, 2.1],
            working_point=0.8,
            disc_cut=None,
            fixed_eff_bin=True,
            marker='o',
            markersize=4,
            is_marker=True,
            label=label,
            colour=colours[i],

        )
        plot_mu = VarVsEff(
            x_var_sig=score_dataset['corrected_averageInteractionsPerCrossing'][is_quark],
            disc_sig=score_dataset[score_name][is_quark],
            weights_sig=score_dataset[args.weight][is_quark] if args.weight is not None else None,
            x_var_bkg=score_dataset['corrected_averageInteractionsPerCrossing'][is_gluon],
            disc_bkg=score_dataset[score_name][is_gluon],
            weights_bkg=score_dataset[args.weight][is_gluon] if args.weight is not None else None,
            # bins=bins,
            bins=[10, 18, 26, 36, 46, 58, 72, 100],
            working_point=0.5,
            disc_cut=None,
            fixed_eff_bin=True,
            marker='o',
            markersize=4,
            is_marker=True,
            label=label,
            colour=colours[i],

        )
        plot_80_mu = VarVsEff(
            x_var_sig=score_dataset['corrected_averageInteractionsPerCrossing'][is_quark],
            disc_sig=score_dataset[score_name][is_quark],
            weights_sig=score_dataset[args.weight][is_quark] if args.weight is not None else None,
            x_var_bkg=score_dataset['corrected_averageInteractionsPerCrossing'][is_gluon],
            disc_bkg=score_dataset[score_name][is_gluon],
            weights_bkg=score_dataset[args.weight][is_gluon] if args.weight is not None else None,
            # bins=bins,
            bins=[10, 18, 26, 36, 46, 58, 72, 100],
            working_point=0.8,
            disc_cut=None,
            fixed_eff_bin=True,
            marker='o',
            markersize=4,
            is_marker=True,
            label=label,
            colour=colours[i],

        )
        
        
        rejs = calc_rej(score_dataset[score_name][is_quark], score_dataset[score_name][is_gluon],
                        sig_eff, score_dataset[args.weight][is_gluon] if args.weight is not None else None)
        plot_roc.add_roc(
            Roc(
                sig_eff,
                rejs,
                n_test=None,

                # rej_class="gluon",
                # signal_class="quark",
                label=label,
                colour=colours[i],
            ),
            reference=True if score_name == 'transformer_score' else False
        )
        plot_bkg_rej.add(plot, reference=True if score_name == 'transformer_score' else False)
        plot_bkg_rej_80.add(plot_80, reference=True if score_name == 'transformer_score' else False)
        plot_bkg_rej_eta.add(plot_eta, reference=True if score_name == 'transformer_score' else False)
        plot_bkg_rej_80_eta.add(plot_80_eta, reference=True if score_name == 'transformer_score' else False)
        plot_bkg_rej_mu.add(plot_mu, reference=True if score_name == 'transformer_score' else False)
        plot_bkg_rej_80_mu.add(plot_80_mu, reference=True if score_name == 'transformer_score' else False)

    plot_bkg_rej.draw()
    plot_bkg_rej_80.draw()
    plot_bkg_rej_eta.draw()
    plot_bkg_rej_80_eta.draw()
    plot_bkg_rej_mu.draw()
    plot_bkg_rej_80_mu.draw()
    plot_roc.draw()
    for ending in ['png', 'pdf']:
        plot_roc.savefig(os.path.join(args.save_path, f'roc.{ending}'), transparent=False, dpi=400)
        plot_bkg_rej.savefig(os.path.join(args.save_path, f'bkg_rej_50.{ending}'), dpi=400)
        plot_bkg_rej_80.savefig(os.path.join(args.save_path, f'bkg_rej_80.{ending}'), dpi=400)
        plot_bkg_rej_eta.savefig(os.path.join(args.save_path, f'bkg_rej_eta_50.{ending}'), dpi=400)
        plot_bkg_rej_80_eta.savefig(os.path.join(args.save_path, f'bkg_rej_eta_80.{ending}'), dpi=400)
        plot_bkg_rej_mu.savefig(os.path.join(args.save_path, f'bkg_rej_mu_50.{ending}'), dpi=400)
        plot_bkg_rej_80_mu.savefig(os.path.join(args.save_path, f'bkg_rej_mu_80.{ending}'), dpi=400)


if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    # original_path = args.save_path
    # for i in range(3, 8):
    #     args.cut = f'JZ_slice == {i}'
    #     args.save_path = f'{original_path}/JZ{i}'
    #     args.title = f'JZ{i}'
    #     args.pt_bins = [200, 300, 400, 600, 850, 1100, 1400, 1750, 2500]
    #     for i in range(len(args.pt_bins)):
    #         try:
    #             main(args)
    #             break
    #         except:
    #             args.pt_bins = args.pt_bins[:-1]
    main(args)
