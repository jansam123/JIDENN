import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from abc import abstractmethod
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, auc
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
parser.add_argument("--summary_csv", default=".", type=str,
                    help="Path to the summary csv.")
parser.add_argument("--model_names", nargs='*', type=str,
                    help="names of the models.")
parser.add_argument("--save_dir", default=".", type=str,
                    help="Directory to save the plots to.")
parser.add_argument("--type", default="pT", type=str, help="Type of the plot.")
parser.add_argument("--compare_mc", default=False, type=bool,
                    help="Compare MC models.")


def main(args: argparse.Namespace):
    metric_names = ["gluon_rejection", "quark_efficiency", 'gluon_efficiency']
    reference = 'pythia'

    [plot_var_dependence(dfs=[pd.read_csv(os.path.join(args.load_dir, f'{mc}', 'models', f'{model}', 'binned_metrics.csv')) for mc in args.mc_names],
                         labels=[MC_NAMING_SCHEMA[mc] for mc in args.mc_names],
                         bin_midpoint_name='bin_mid',
                         bin_width_name='bin_width',
                         metric_names=metric_names,
                         save_path=os.path.join(args.save_dir, 'models', f'{model}'),
                         ratio_reference_label=MC_NAMING_SCHEMA[reference],
                         xlabel=r'$p_T$ [TeV]',
                         ylabel_mapper=METRIC_NAMING_SCHEMA,
                         ylims=[(5, 30), (0.1, 1.), (0.1, 1.)],
                         xlog=True,
                         h_line_position=[None, 0.5, None],
                         leg_loc='upper center',
                         title=f'{MODEL_NAMING_SCHEMA[model]}',
                         colours=sns.color_palette("Set1", len(args.mc_names))) for model in args.model_names]

    metric = 'quark_efficiency'
    df_big = pd.DataFrame()

    for mc in args.mc_names:
        overall_metrics = pd.read_csv(os.path.join(args.load_dir, f'{mc}', 'overall_metrics.csv'), index_col=0)
        acc_sorted_models = overall_metrics.sort_values(by='binary_accuracy', ascending=False).index
        dfs = []
        diff_name = 'diff'
        for label in args.model_names:
            ref_df = pd.read_csv(os.path.join(
                args.load_dir, f'{reference}', 'models', f'{label}', 'binned_metrics.csv'))
            if label == reference:
                continue
            df = pd.read_csv(os.path.join(args.load_dir, f'{mc}', 'models', f'{label}', 'binned_metrics.csv'))
            df['ratio'] = df[metric] / ref_df[metric]
            df[diff_name] = ref_df[metric] - df[metric]
            dfs.append(df)
            df['mc'] = mc
            df['model'] = label
            df_big = pd.concat([df_big, df])

        labels, dfs = zip(*sorted(zip(args.model_names, dfs), key=lambda x: acc_sorted_models.get_loc(x[0])))
        labels = [MODEL_NAMING_SCHEMA[model] for model in labels]

        plot_var_dependence(dfs=dfs,
                            labels=labels,
                            bin_midpoint_name='bin_mid',
                            bin_width_name='bin_width',
                            metric_names=[diff_name, 'ratio'],
                            save_path=os.path.join(args.save_dir, 'MCs', f'{mc}'),
                            ratio_reference_label=None,
                            xlabel=r'$p_T$ [TeV]',
                            ylabel_mapper={diff_name: f'Difference fromn Pythia', 'ratio': f'Ratio'},
                            ylims=[(-0.1, 0.175), (0.7, 1.3)],
                            # ylims = None,
                            xlog=True,
                            figsize=(10, 4),
                            h_line_position=[0.0, 1.0],
                            leg_loc='upper center',
                            title=f'{MC_NAMING_SCHEMA[mc]}',
                            colours=sns.color_palette("coolwarm", len(args.model_names)))

    df_big = df_big[['mc', 'model', metric]]
    df_big = df_big.groupby(['mc', 'model']).mean().reset_index()
    df_big['model'] = df_big['model'].map(MODEL_NAMING_SCHEMA)
    df_big['mc'] = df_big['mc'].map(MC_NAMING_SCHEMA)
    df_big = df_big.rename(columns={'mc': 'MC Sample', 'model': 'Deep Learning Model'})
    fig = plt.figure(figsize=(14, 6))
    ax = sns.histplot(data=df_big, hue='MC Sample', x='Deep Learning Model',
                      multiple='dodge', palette='Set1', weights=metric,
                      shrink=.8, hue_order=[MC_NAMING_SCHEMA[mc] for mc in args.mc_names])

    ax.legend_.set_title(None)
    plt.ylim(0., 0.8)
    plt.ylabel(METRIC_NAMING_SCHEMA[metric])
    atlasify.atlasify(
        atlas=True,
        subtext="Simulation Internal \n 13 TeV",
    )
    plt.savefig(os.path.join(args.save_dir, f'summary_hist.png'), dpi=400, bbox_inches='tight')


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    args.load_dir = '/home/jankovys/JIDENN/logs/pythia_log_flat_70_JZ7/evaluation'
    args.save_dir = '/home/jankovys/JIDENN/logs/pythia_log_flat_70_JZ7/evaluation/compare_mc'
    args.mc_names = ['pythia', 'herwig7_dipole', 'sherpa_dire', 'herwig7',
                     'sherpa_enh_cluster_tune', 'sherpa', 'sherpa_lund', 'powheg_pythia']
    args.model_names = ["idepart", "ipart", "depart",
                        "part", "transformer", "efn", "pfn", "fc", "highway"]
    main(args)
