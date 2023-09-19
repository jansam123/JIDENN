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
parser.add_argument("--model_names", nargs='*', type=str,
                    help="names of the models.")
parser.add_argument("--save_dir", default=".", type=str,
                    help="Directory to save the plots to.")
parser.add_argument("--var", default="jets_pt", type=str,
                    help="Variable to plot.")
parser.add_argument("--type", default="pT", type=str, help="Type of the plot.")
parser.add_argument("--compare_mc", default=False, type=bool,
                    help="Compare MC models.")


def compare_ml_models(overall_metrics_path: str,
                      paths: List[str],
                      labels: List[str],
                      save_path: str,
                      x_var = 'jets_pt',):

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
    if x_var == 'jets_pt':
        for df in sorted_dfs:
            df['bin_mid'] = df['bin_mid'] * 1e-6
            df['bin_width'] = df['bin_width'] * 1e-6

    plot_var_dependence(dfs=sorted_dfs,
                        labels=[MODEL_NAMING_SCHEMA[model] for model in list(sorted_labels)],
                        bin_midpoint_name='bin_mid',
                        bin_width_name='bin_width',
                        metric_names=metric_names,
                        save_path=save_path,
                        ratio_reference_label=None,
                        xlabel=LATEX_NAMING_CONVENTION[x_var],
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
                      save_path=args.save_dir,
                      x_var=args.var)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    # args.load_dir = 'logs/stepwise_flat/eval'
    # args.save_dir = 'plots/stepwise_flat/eval/post_compare_models'
    # args.model_names = ["idepart", "ipart", "depart", "particle_net",
    #                     "part", "transformer", "efn", "pfn", "fc", "highway"]
    args.model_names = ["idepart", "ipart", "particle_net", "pfn", "efn", "fc", "highway"]
    main(args)
