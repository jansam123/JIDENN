import os
import sys
sys.path.append(os.getcwd())
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np
from io import BytesIO
from typing import List, Union, Dict, Optional, Tuple
import atlasify
import argparse
import puma
#
from jidenn.evaluation.plotter import plot_var_dependence
from jidenn.const import METRIC_NAMING_SCHEMA, LATEX_NAMING_CONVENTION, MODEL_NAMING_SCHEMA, MC_NAMING_SCHEMA

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--load_dir", default=".", type=str,
                    help="Directory to load the metrics from.")
parser.add_argument("-m", "--model_names", nargs='*', type=str,
                    help="names of the models.")
parser.add_argument("--save_dir", default=".", type=str,
                    help="Directory to save the plots to.")
parser.add_argument("--var", default="jets_pt", type=str,
                    help="Variable to plot.")
parser.add_argument("--type", default="pT", type=str, help="Type of the plot.")
parser.add_argument("--compare_mc", default=False, type=bool,
                    help="Compare MC models.")
parser.add_argument("-f", "--figsize", default=(14, 9), type=float, nargs=2,  # (13, 9)
                    help="Figure size.")
parser.add_argument("--leg_ncol", default=2, type=int,
                    help="Number of columns in the legend.")
parser.add_argument("--mc_type", default="Pythia8EvtGen_A14NNPDF23LO_jetjet", type=str,
                    help="MC type.")
parser.add_argument("--release", default="r22", type=str,
                    help="MC type.")


def compare_ml_models(paths: List[str],
                      labels: List[str],
                      save_path: str,
                      figsize: Tuple[int, int] = (9, 5),
                      leg_ncol: int = 2,
                      x_var='jets_pt',):

    os.makedirs(save_path, exist_ok=True)
    dfs = [pd.read_csv(path) for path in paths]
    accuracies = [df['gluon_rejection_at_quark_80wp'].mean() for df in dfs]
    # labels, dfs, accuracies = zip(*sorted(zip(labels, dfs, accuracies),
    #                                                     key=lambda x: x[2], reverse=True))
    for label, acc in zip(labels, accuracies):
        print(f'{label}: {acc:.4f}')
    pd.DataFrame({'Model': labels, 'Accuracy': accuracies}).to_csv(
        os.path.join(save_path, 'sorted_accuracies.csv'), index=False)
    print(labels)
    metric_names = ["gluon_efficiency", "quark_efficiency",
                    "gluon_rejection", "quark_rejection",
                    "binary_accuracy", "auc",
                    # "effective_tagging_efficiency",
                    'gluon_rejection_at_quark_80wp', 'gluon_efficiency_at_quark_80wp',
                    'gluon_rejection_at_quark_50wp', 'gluon_efficiency_at_quark_50wp']
    n_counts = ['eff_num_events_g', 'eff_num_events_q',
                'eff_num_events_g', 'eff_num_events_q',
                'eff_num_events', 'eff_num_events',
                # 'eff_num_events_g',
                'eff_num_events_g', 'eff_num_events_g',
                'eff_num_events_g', 'eff_num_events_g']

    # ylims = [[0.6, 0.9], [0.55, 0.9], [0.6, 0.85], [0.7, 0.9], [0.2, 0.5],
    #          [2.5, 6], [3, 35], [0.53, 0.85], [0.85, 1.0]]
    if x_var == 'jets_pt':
        ylims = [None]*6 + [[2., 6.8], None, [6.5, 36.], None]
    elif x_var == 'jets_eta':
        ylims = [None]*6 + [[1., 4.8], None, [3., 22.5], None]
    colours = sns.color_palette('colorblind', len(labels))
    # if x_var == 'jets_pt':
    #     for df in dfs:
    #         df['bin_mid'] = df['bin_mid'] * 1e-6
    #         df['bin_width'] = df['bin_width'] * 1e-6
    if x_var == 'jets_eta':
        for df in dfs:
            df = df[df['bin_mid'] > 0]
    mc_label = MC_NAMING_SCHEMA[args.mc_type]
    title_50 = mc_label+', 50% WP\n'
    title_80 = mc_label+', 80% WP\n'
    title_none = mc_label+'\n'
    if args.release == 'r21':
        title_all = r'anti-$k_{\mathrm{T}}$, $R = 0.4$ PFlow jets'
    elif args.release == 'r22':
        title_all = r'anti-$k_{\mathrm{T}}$, $R = 0.4$ UFO CSSK jets'
    else:
        raise ValueError('Release not supported')
    new_labels = []
    for label in labels:
        new_labels.append(
            MODEL_NAMING_SCHEMA[label] if label in MODEL_NAMING_SCHEMA else label)
    plot_var_dependence(dfs=dfs,
                        labels=new_labels,
                        bin_midpoint_name='bin_mid',
                        bin_width_name='bin_width',
                        n_counts=n_counts,
                        metric_names=metric_names,
                        save_path=save_path,
                        ratio_reference_label=None,
                        xlabel=LATEX_NAMING_CONVENTION[x_var] if x_var in LATEX_NAMING_CONVENTION else x_var,
                        ylabel_mapper=METRIC_NAMING_SCHEMA,
                        ylims=ylims,
                        xlims=None,
                        xlog=False,
                        title=[title_none + title_all]*6 + [title_80 +
                                                            title_all]*2 + [title_50 + title_all]*2,
                        figsize=figsize,
                        leg_loc='upper right',
                        leg_ncol=leg_ncol,
                        colours=colours,
                        label_fontsize=30,
                        fontsize=24,
                        leg_fontsize=24,
                        atlas_fontsize=24,
                        )


def main(args: argparse.Namespace):

    paths = []
    for model in args.model_names:
        if not os.path.exists(os.path.join(args.load_dir, f'{model}', 'binned_metrics.csv')):
            print(f'No binned metrics for {model}')
            continue
        paths += [os.path.join(args.load_dir,
                               f'{model}', 'binned_metrics.csv')]

    compare_ml_models(paths=paths,
                      labels=args.model_names,
                      save_path=args.save_dir,
                      figsize=(args.figsize[0], args.figsize[1]),
                      leg_ncol=args.leg_ncol,
                      x_var=args.var)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    # args.load_dir = 'logs/stepwise_flat/eval'
    # args.save_dir = 'plots/stepwise_flat/eval/post_compare_models'
    # args.model_names = ["idepart", "ipart", "depart", "particle_net",
    # "part", "transformer", "efn", "pfn", "fc", "highway", "fc_crafted", "highway_crafted"]
    # if args.model_names is None:
    for var in ['eta', 'pt']:
        args.load_dir = f'logs/r22_forward_lead_sublead/evaluation/pythia_nominal-{var}/models'
        args.save_dir = f'logs/r22_forward_lead_sublead/evaluation/pythia_nominal-{var}/plots'
        args.var = 'jets_'+var
        # args.model_names = ["idepart", "idepart+topo", "ipart",
        #                     "ipart+topo", "pfn", "pfn+topo",]# "fc", "fc_crafted", "efn"]
        args.model_names = ["idepart", "idepart+topo",
                            "pfn", "pfn+topo", "fc", "fc_crafted", "efn", "efn+topo"]
        main(args)
    # args.model_names = ["depart", "depart-multiMC"]
    # args.model_names = ["depart", "depart-multiMC"]
    # args.model_names = ["fc", "highway","fc_crafted", "highway_crafted"]
    # path = '/home/jankovys/JIDENN/logs/augmentations/evaluation/50wp'
    # for mc_name in os.listdir(path):
    #     # check if the directory is a MC type
    #     if mc_name not in MC_NAMING_SCHEMA:
    #         continue
    #     args.mc_type = mc_name
    #     args.load_dir = os.path.join(path, mc_name, 'models')
    #     args.save_dir = os.path.join(path, mc_name, 'plots2')
    #     # args.model_names = ['depart', 'depart-all_aug', 'depart-coll_split', 'depart-pt_smear', 'depart-rot_drop_smear', 'depart-rotation',
    #     #                     'depart-shift_weights', 'depart-soft_drop', 'depart-soft_smear', 'pfn', 'fc_crafted', 'depart-ircs', 'efn']
    #     args.model_names = ['depart', 'depart-Sh_2211_Enh_clusterTune', 'depart-H7EG_jetjet_Cluster',
    #                         'depart-H7EG_jetjet_Cluster_dipole', 'depart-Sherpa_CT10_CT14nnlo_CSShower_Lund_2to2jets', 'depart-Sh_2211_jj_DIRE', 'depart-multiMC', 'pfn']

    #     main(args)
