import pandas as pd
import os
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
from jidenn.const import METRIC_NAMING_SCHEMA, LATEX_NAMING_CONVENTION, MODEL_NAMING_SCHEMA

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--load_dir", default=".", type=str,
                    help="Directory to load the metrics from.")
parser.add_argument("--summary_csv", default=".", type=str,
                    help="Path to the summary csv.")
parser.add_argument("-m", "--model_names", nargs='*', type=str,
                    help="names of the models.")
parser.add_argument("--save_dir", default=".", type=str,
                    help="Directory to save the plots to.")
parser.add_argument("--var", default="jets_pt", type=str,
                    help="Variable to plot.")
parser.add_argument("--type", default="pT", type=str, help="Type of the plot.")
parser.add_argument("--compare_mc", default=False, type=bool,
                    help="Compare MC models.")
parser.add_argument("-f", "--figsize", default=(7, 5), type=float, nargs=2,
                    help="Figure size.")
parser.add_argument("--leg_ncol", default=2, type=int,
                    help="Number of columns in the legend.")

def compare_ml_models(paths: List[str],
                      labels: List[str],
                      save_path: str,
                      figsize: Tuple[int, int] = (7, 5),
                      leg_ncol: int = 2,
                      x_var = 'jets_pt',):

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
                    'gluon_rejection_at_quark_50wp','gluon_efficiency_at_quark_50wp']
    n_counts = ['eff_num_events_g', 'eff_num_events_q',
                'eff_num_events_g', 'eff_num_events_q',
                'eff_num_events', 'eff_num_events',
                # 'eff_num_events_g',
                'eff_num_events_g', 'eff_num_events_g',
                'eff_num_events_g', 'eff_num_events_g']

    # ylims = [[0.6, 0.9], [0.55, 0.9], [0.6, 0.85], [0.7, 0.9], [0.2, 0.5],
    #          [2.5, 6], [3, 35], [0.53, 0.85], [0.85, 1.0]]
    ylims = [None]*6 + [[2., 6.8], None, [6.5, 36.], None]
    reference = 'highway'
    colours = sns.color_palette('colorblind', len(labels))
    if x_var == 'jets_pt':
        for df in dfs:
            df['bin_mid'] = df['bin_mid'] * 1e-6
            df['bin_width'] = df['bin_width'] * 1e-6
    if x_var == 'jets_eta':
        for df in dfs:
            df = df[df['bin_mid'] > 0]
            
    title_50 = 'Pythia8, 50% WP\n'
    title_80 = 'Pythia8, 80% WP\n'
    title_none = 'Pythia8\n'
    title_all = r'anti-$k_{\mathrm{T}}$, $R = 0.4$ PFlow jets'
    plot_var_dependence(dfs=dfs,
                        labels=[MODEL_NAMING_SCHEMA[model] for model in list(labels)],
                        bin_midpoint_name='bin_mid',
                        bin_width_name='bin_width',
                        n_counts = n_counts,
                        metric_names=metric_names,
                        save_path=save_path,
                        ratio_reference_label=None,
                        xlabel=LATEX_NAMING_CONVENTION[x_var],
                        ylabel_mapper=METRIC_NAMING_SCHEMA,
                        ylims=ylims,
                        xlog=False,
                        title=[title_none + title_all]*6 +  [title_80 + title_all]*2 + [title_50 + title_all]*2,
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
        paths += [os.path.join(args.load_dir, f'{model}', 'binned_metrics.csv')]

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
    if args.model_names is None:
        args.model_names = ["idepart", "ipart", "particle_net", "pfn", "efn", "fc", "highway"]
    # args.model_names = ["fc", "highway","fc_crafted", "highway_crafted"]
    main(args)
