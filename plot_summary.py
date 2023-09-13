import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from abc import abstractmethod
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, auc
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
parser.add_argument("--model_names", nargs='*', type=str,
                    help="names of the models.")
parser.add_argument("--save_dir", default=".", type=str,
                    help="Directory to save the plots to.")
parser.add_argument("--type", default="pT", type=str, help="Type of the plot.")
parser.add_argument("--compare_mc", default=False, type=bool,
                    help="Compare MC models.")


def plot_metric(df: pd.DataFrame,
                relative_df: pd.DataFrame,
                metric: str,
                x_label: str,
                save_dir: str,
                title: str,
                order: Optional[List[str]] = None,
                ylim: Optional[Tuple[float, float]] = None,):

    palette = 'coolwarm'
    fig_big = plt.figure(figsize=(16, 10))
    gs = fig_big.add_gridspec(2, hspace=0.07, height_ratios=[2.5, 1])
    ax1, ax2 = gs.subplots(sharex=True, sharey=False)
    sns.pointplot(x='cut', y=metric, data=df, hue='Model', errorbar=None,
                    palette=palette, hue_order=order, ax=ax1)

    ax1.set(ylabel=METRIC_NAMING_SCHEMA[metric]
            if metric in METRIC_NAMING_SCHEMA else metric, xlabel=None)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles, labels=labels, loc='upper right', ncol=2)

    sns.pointplot(x='cut', y=metric, data=relative_df, hue='Model', errorbar=None,
                    palette=palette, hue_order=order, ax=ax2, estimator=np.mean)
    ax2.set(ylabel='Ratio', xlabel=x_label)
    ax2.get_legend().set_visible(False)
    if ylim is not None:
        ax1.set_ylim(ylim)

    # make a gap between the two plots
    atlasify.atlasify("Simulation Internal", axes=ax1, subtext='13 TeV')
    atlasify.atlasify(atlas=False, axes=ax2)

    fig_big.savefig(save_dir, bbox_inches='tight', dpi=400)
    plt.close(fig_big)


def compare_ml_models(overall_metrics_path: str,
                      paths: List[str],
                      labels: List[str],
                      save_path: str,):

    os.makedirs(save_path, exist_ok=True)
    dfs = [pd.read_csv(path) for path in paths]
    accuracies = [df['gluon_rejection_at_quark_80wp'].mean() for df in dfs]
    sorted_labels, sorted_dfs, accuracies = zip(*sorted(zip(labels, dfs, accuracies),
                                                        key=lambda x: x[2], reverse=True))
    for label, acc in zip(sorted_labels, accuracies):
        print(f'{label}: {acc:.4f}')
    pd.DataFrame({'Model': sorted_labels, 'Accuracy': accuracies}).to_csv(
        os.path.join(save_path, 'sorted_accuracies.csv'), index=False)
    print(sorted_labels)
    metric_names = ["gluon_efficiency", "quark_efficiency",
                    "binary_accuracy", "auc",
                    # "effective_tagging_efficiency",
                    'gluon_rejection_at_quark_80wp', 'gluon_rejection_at_quark_50wp',
                    'gluon_efficiency_at_quark_80wp', 'gluon_efficiency_at_quark_50wp']

    # ylims = [[0.6, 0.9], [0.55, 0.9], [0.6, 0.85], [0.7, 0.9], [0.2, 0.5],
    #          [2.5, 6], [3, 35], [0.53, 0.85], [0.85, 1.0]]
    ylims = None
    reference = 'highway'
    colours = sns.color_palette('coolwarm', len(sorted_labels))

    plot_var_dependence(dfs=sorted_dfs,
                        labels=[MODEL_NAMING_SCHEMA[model] for model in list(sorted_labels)],
                        bin_midpoint_name='bin_mid',
                        bin_width_name='bin_width',
                        metric_names=metric_names,
                        save_path=save_path,
                        ratio_reference_label=None,#MODEL_NAMING_SCHEMA[reference],
                        xlabel=r'$p_T$ [TeV]', 
                        ylabel_mapper=METRIC_NAMING_SCHEMA,
                        ylims=ylims,
                        xlog=False,
                        figsize=(7, 5),
                        leg_loc='upper right',
                        colours=colours)


def main(args: argparse.Namespace):

    overall_metrics_path = args.summary_csv
    paths = []
    for model in args.model_names:
        if not os.path.exists(os.path.join(args.load_dir, f'{model}', 'binned_metrics.csv')):
            print(f'No binned metrics for {model}')
            continue
        paths += [os.path.join(args.load_dir, f'{model}', 'binned_metrics.csv')]

    compare_ml_models(overall_metrics_path=overall_metrics_path,
                      paths=paths,
                      labels=args.model_names,
                      save_path=args.save_dir)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    # args.load_dir = 'logs/stepwise_flat/eval'
    # args.save_dir = 'plots/stepwise_flat/eval/post_compare_models'
    args.model_names = ["idepart", "ipart", "depart", "particle_net",
                        "part", "transformer", "efn", "pfn", "fc", "highway"]
    # args.model_names = ["idepart", "depart", "particle_net", "pfn", "highway"]
    main(args)
