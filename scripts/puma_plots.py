import os
import sys
sys.path.append(os.getcwd())
from puma import VarVsEff, VarVsEffPlot, RocPlot, Roc
from puma.metrics import calc_rej
import pandas as pd
import numpy as np
import seaborn as sns
import argparse
from sklearn.metrics import roc_curve, precision_recall_curve, precision_score

# from jidenn.config.eval_config import BinningConfig
from jidenn.const import MODEL_NAMING_SCHEMA
from jidenn.evaluation.evaluation_metrics import RejectionAtFixedWorkingPoint, BkgRejVsSigEff

parser = argparse.ArgumentParser()
parser.add_argument("--load_path", type=str,
                    help="Path to the saved tf.data.Dataset file")
parser.add_argument("--save_path", type=str, default='plots',
                    help="Path to save the plots")
parser.add_argument("--models", type=str, nargs='+',
                    help="Models to compare")
parser.add_argument("-v", "--variable", type=str,
                    default='jets_pt', help="Variables to plot")
parser.add_argument("-w", "--weight", type=str,
                    default='weight', help="Weight variable")
parser.add_argument("--bins", type=int, default=50,
                    help="Number of bins for the histograms")
parser.add_argument("--log_bins", action='store_true',
                    help="Whether to use log-spaced bins")
parser.add_argument("--ylog", action='store_true',
                    help="Whether to use log-spaced bins")
parser.add_argument("--xlog", action='store_true',
                    help="Whether to use log-spaced bins")
parser.add_argument("--no_badge", action='store_true',
                    help="Whether to use log-spaced bins")
parser.add_argument("--xlim", type=float, nargs=2,
                    default=None, help="X-axis limits")
parser.add_argument("--ylim", type=float, nargs=2,
                    default=None, help="Y-axis limits")
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

parser.add_argument("--release", type=str, default='r22',
                    help="MC release")

HUE_MAPPER = {1: 'quark', 2: 'quark', 3: 'quark',
              4: 'quark', 5: 'quark', 6: 'quark', 21: 'gluon'}


def main(args):

    # score_dataset = pd.read_csv(args.load_path, index_col=0)

    # if args.scores is None:
    #     args.scores = [col for col in score_dataset.columns if 'score' in col]
    # if args.cut is not None else score_dataset

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
    if args.title is not None:
        tag = args.title
    else:
        tag = '13 TeV, Pythia8\n' 
        tag += r"anti-$k_{\mathrm{T}}$, $R = 0.4$ PFlow jets" if args.release == 'r21' else r"anti-$k_{\mathrm{T}}$, $R = 0.4$  "
        
    plot_roc = RocPlot(
        n_ratio_panels=1,
        ylabel=r"$\varepsilon_g^{-1}$",
        xlabel=r"$\varepsilon_q$",
        atlas_second_tag=tag,
        atlas_first_tag="Simulation Internal",
        figsize=(8, 8),
        ymin=1,
        ymax=1e3,
        xmin=0.1,
        xmax=1,
        y_scale=1.4,
        grid=False,
        atlas_fontsize=15,
        leg_fontsize=15,
        fontsize=17,
        label_fontsize=18,
    )

    plots = []
    sig_eff = np.linspace(0.1, 1, 100)
    # score_dataset.columns:
    # scores = ["idepart_score", "ipart_score", "particle_net_score", "depart_score",
    #           "transformer_score", "part_score", "pfn_score", "highway_score", "fc_score", "efn_score",]

    colours = sns.color_palette('colorblind', len(args.models))
    lines = ['-', '--', '-.', ':',
             (5, (10, 3)), (0, (3, 5, 1, 5)), (0, (5, 1))]
    rej_calculators = [RejectionAtFixedWorkingPoint(name='gluon_rejection_at_quark_50wp',
                                                    fixed_label_id=1,
                                                    working_point=wp,
                                                    num_thresholds=200,
                                                    returned_label_id=0) for wp in sig_eff]
    print(args.weight)
    for i, model in enumerate(args.models):
        score_dataset = pd.read_csv(os.path.join(
            args.load_path, model, 'score_dataset.csv'), index_col=0)
        score_dataset = score_dataset.query(
            'jets_pt < 2500000 and jets_pt > 200000 and jets_eta < 4.1 and jets_eta > -4.1')
        score_dataset['jets_pt'] = score_dataset['jets_pt'] * 1e-3
        is_quark = score_dataset['label'] == 1
        is_gluon = score_dataset['label'] == 0
        score_dataset['jets_eta'] = score_dataset['jets_eta'].abs()
        score_name = f'{model}_score'
        print(score_dataset["corrected_averageInteractionsPerCrossing"])
        print(score_dataset["jets_pt"])
        print(score_dataset["jets_eta"])
        print(score_dataset[score_name])
        label = MODEL_NAMING_SCHEMA[model] if model in MODEL_NAMING_SCHEMA.keys(
        ) else model
        plot = VarVsEff(
            x_var_sig=score_dataset['jets_pt'][is_quark],
            disc_sig=score_dataset[score_name][is_quark],
            weights_sig=score_dataset[args.weight][is_quark] if args.weight is not None else None,
            x_var_bkg=score_dataset['jets_pt'][is_gluon],
            disc_bkg=score_dataset[score_name][is_gluon],
            weights_bkg=score_dataset[args.weight][is_gluon] if args.weight is not None else None,
            # bins=bins,
            bins=args.pt_bins if args.pt_bins is not None else [
                200, 300, 400, 600, 850, 1100, 1400, 1750, 2500],
            working_point=0.5,
            disc_cut=None,
            fixed_eff_bin=True,
            marker='o',
            markersize=4,
            is_marker=True,
            label=label,
            colour=colours[i % len(colours)],

        )
        plot_80 = VarVsEff(
            x_var_sig=score_dataset['jets_pt'][is_quark],
            disc_sig=score_dataset[score_name][is_quark],
            weights_sig=score_dataset[args.weight][is_quark] if args.weight is not None else None,
            x_var_bkg=score_dataset['jets_pt'][is_gluon],
            disc_bkg=score_dataset[score_name][is_gluon],
            weights_bkg=score_dataset[args.weight][is_gluon] if args.weight is not None else None,
            # bins=bins,
            bins=args.pt_bins if args.pt_bins is not None else [
                200, 300, 400, 600, 850, 1100, 1400, 1750, 2500],
            working_point=0.8,
            disc_cut=None,
            fixed_eff_bin=True,
            marker='o',
            markersize=4,
            is_marker=True,
            label=label,
            colour=colours[i % len(colours)],

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
            colour=colours[i % len(colours)],

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
            colour=colours[i % len(colours)],

        )
        # plot_mu = VarVsEff(
        #     x_var_sig=score_dataset['corrected_averageInteractionsPerCrossing'][is_quark],
        #     disc_sig=score_dataset[score_name][is_quark],
        #     weights_sig=score_dataset[args.weight][is_quark] if args.weight is not None else None,
        #     x_var_bkg=score_dataset['corrected_averageInteractionsPerCrossing'][is_gluon],
        #     disc_bkg=score_dataset[score_name][is_gluon],
        #     weights_bkg=score_dataset[args.weight][is_gluon] if args.weight is not None else None,
        #     # bins=bins,
        #     bins=[10, 18, 26, 36, 46, 58, 72, 100],
        #     working_point=0.5,
        #     disc_cut=None,
        #     fixed_eff_bin=True,
        #     marker='o',
        #     markersize=4,
        #     is_marker=True,
        #     label=label,
        #     colour=colours[i % len(colours)],

        # )
        # plot_80_mu = VarVsEff(
        #     x_var_sig=score_dataset['corrected_averageInteractionsPerCrossing'][is_quark],
        #     disc_sig=score_dataset[score_name][is_quark],
        #     weights_sig=score_dataset[args.weight][is_quark] if args.weight is not None else None,
        #     x_var_bkg=score_dataset['corrected_averageInteractionsPerCrossing'][is_gluon],
        #     disc_bkg=score_dataset[score_name][is_gluon],
        #     weights_bkg=score_dataset[args.weight][is_gluon] if args.weight is not None else None,
        #     # bins=bins,
        #     bins=[10, 18, 26, 36, 46, 58, 72, 100],
        #     working_point=0.8,
        #     disc_cut=None,
        #     fixed_eff_bin=True,
        #     marker='o',
        #     markersize=4,
        #     is_marker=True,
        #     label=label,
        #     colour=colours[i % len(colours)],

        # )

        # rejs = calc_rej(score_dataset[score_name][is_quark], score_dataset[score_name][is_gluon],
        # sig_eff)#, bkg_weights=score_dataset[args.weight][is_gluon] if args.weight is not None else None)
        # rejs = [rc(y_true=score_dataset['label'],
        #            y_pred=score_dataset[score_name],
        #            sample_weight=score_dataset[args.weight]).numpy() for rc in rej_calculators]
        # rejs = np.array(rejs)
        sig_eff, rejs = BkgRejVsSigEff(
            thresholds=list(np.linspace(0., 1, 200)),
            fixed_label_id=1)(y_true=score_dataset['label'],
                              y_pred=score_dataset[score_name],
                              sample_weight=score_dataset[args.weight])
        sig_eff = sig_eff.numpy()
        sig_eff, uniq_idx = np.unique(sig_eff, return_index=True)
        rejs = rejs.numpy()
        rejs = rejs[uniq_idx]
        rejs = rejs[np.where(sig_eff > 0.04)]
        sig_eff = sig_eff[np.where(sig_eff > 0.04)]

        sort_idx = np.argsort(sig_eff)
        rejs = rejs[sort_idx]
        sig_eff = sig_eff[sort_idx]
        print(sig_eff)
        print(rejs)
        plot_roc.add_roc(
            Roc(
                sig_eff,
                rejs,
                n_test=None,
                rej_class="ujets",
                # signal_class="quark",
                label=label,
                colour=colours[i % len(colours)],
                linestyle=lines[i % len(lines)],
            ),
            reference=True if score_name == args.ref_score else False
        )
        plot_bkg_rej.add(plot, reference=True if score_name ==
                         'transformer_score' else False)
        plot_bkg_rej_80.add(
            plot_80, reference=True if score_name == 'transformer_score' else False)
        plot_bkg_rej_eta.add(
            plot_eta, reference=True if score_name == 'transformer_score' else False)
        plot_bkg_rej_80_eta.add(
            plot_80_eta, reference=True if score_name == 'transformer_score' else False)
        # plot_bkg_rej_mu.add(
        #     plot_mu, reference=True if score_name == 'transformer_score' else False)
        # plot_bkg_rej_80_mu.add(
        #     plot_80_mu, reference=True if score_name == 'transformer_score' else False)

    plot_roc.set_ratio_class(1, "ujets")
    plot_bkg_rej.draw()
    plot_bkg_rej_80.draw()
    plot_bkg_rej_eta.draw()
    plot_bkg_rej_80_eta.draw()
    # plot_bkg_rej_mu.draw()
    # plot_bkg_rej_80_mu.draw()
    plot_roc.draw()
    os.makedirs(args.save_path, exist_ok=True)
    for ending in ['png', 'pdf']:
        plot_roc.savefig(os.path.join(
            args.save_path, f'roc.{ending}'), transparent=False, dpi=400)
        plot_bkg_rej.savefig(os.path.join(
            args.save_path, f'bkg_rej_50.{ending}'), dpi=400)
        plot_bkg_rej_80.savefig(os.path.join(
            args.save_path, f'bkg_rej_80.{ending}'), dpi=400)
        plot_bkg_rej_eta.savefig(os.path.join(
            args.save_path, f'bkg_rej_eta_50.{ending}'), dpi=400)
        plot_bkg_rej_80_eta.savefig(os.path.join(
            args.save_path, f'bkg_rej_eta_80.{ending}'), dpi=400)
        # plot_bkg_rej_mu.savefig(os.path.join(
        #     args.save_path, f'bkg_rej_mu_50.{ending}'), dpi=400)
        # plot_bkg_rej_80_mu.savefig(os.path.join(
        #     args.save_path, f'bkg_rej_mu_80.{ending}'), dpi=400)


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
    # args.scores = ["idepart-m_score", "ipart-m_score", "particle_net-m_score",
    #           "pfn-m_score", "fc-reduced_score", "fc_crafted_score", "efn_score",]
    args.models = ["idepart", "idepart+topo",
                        "pfn", "pfn+topo", "fc", "fc_crafted", "efn", "efn+topo"]
    args.load_path = 'logs/r22_forward_lead_sublead/evaluation/pythia_nominal-pt/models'
    args.save_path = 'logs/r22_forward_lead_sublead/evaluation/pythia_nominal-pt/puma_plots'

    # args.models = ["idepart", "ipart", "pfn",  "fc", "fc_crafted", "efn"]
    # args.load_path = 'logs/r22_central_all/evaluation/pythia_nominal-pt/models'
    # args.save_path = 'logs/r22_central_all/evaluation/pythia_nominal-pt/puma_plots'

    args.scores = None
    args.ref_score = "fc_score"
    main(args)
    print('Done')
