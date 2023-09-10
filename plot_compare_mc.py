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

    [plot_var_dependence(dfs=[pd.read_csv(os.path.join(dir, 'models', f'{model}', 'binned_metrics.csv')) for dir in args.load_dirs],
                         labels=args.labels,
                         bin_midpoint_name='bin_mid',
                         bin_width_name='bin_width',
                         metric_names=args.metric_names,
                         save_path=os.path.join(args.save_dir, 'models', f'{model}'),
                         ratio_reference_label=args.reference,
                         xlabel=r'$p_T$ [TeV]',
                         ylabel_mapper=METRIC_NAMING_SCHEMA,
                         ylims=args.ylims1,
                         xlog=args.xlog,
                         h_line_position=[None, 0.5, None],
                         leg_loc='upper center',
                         title=f'{MODEL_NAMING_SCHEMA[model]}',
                         colours=sns.color_palette("Set1", len(args.labels))) for model in args.model_names]

    metric = 'quark_efficiency'
    df_big = pd.DataFrame()

    overall_metrics = pd.read_csv(os.path.join(args.ref_dir, 'overall_metrics.csv'), index_col=0)
    acc_sorted_models = overall_metrics.sort_values(by='binary_accuracy', ascending=False).index
    
    for label, dir in zip(args.labels, args.load_dirs):
        dfs = []
        diff_name = 'diff'
        for model in args.model_names:
            ref_df = pd.read_csv(os.path.join(args.ref_dir, 'models', f'{model}', 'binned_metrics.csv'))
            if model == args.reference:
                continue
            df = pd.read_csv(os.path.join(dir, 'models', f'{model}', 'binned_metrics.csv'))
            df['ratio'] = df[metric] / ref_df[metric]
            df[diff_name] = ref_df[metric] - df[metric]
            dfs.append(df)
            df['mc'] = label
            df['model'] = model
            df_big = pd.concat([df_big, df])

        labels, dfs = zip(*sorted(zip(args.model_names, dfs), key=lambda x: acc_sorted_models.get_loc(x[0])))
        labels = [MODEL_NAMING_SCHEMA[model] for model in labels]

        plot_var_dependence(dfs=dfs,
                            labels=labels,
                            bin_midpoint_name='bin_mid',
                            bin_width_name='bin_width',
                            metric_names=[diff_name, 'ratio'],
                            save_path=os.path.join(args.save_dir, 'MCs', f'{label}'),
                            ratio_reference_label=None,
                            xlabel=r'$p_T$ [TeV]',
                            ylabel_mapper={diff_name: f'Difference fromn Pythia', 'ratio': f'Ratio'},
                            ylims=args.ylims2, 
                            # ylims = None,
                            xlog=args.xlog,
                            figsize=(10, 4),
                            h_line_position=[0.0, 1.0],
                            leg_loc='upper center',
                            title=label,
                            colours=sns.color_palette("coolwarm", len(args.model_names)))

    df_envel = df_big.copy()[['model', 'bin_mid', 'bin_width', 'ratio', diff_name]]
    df_envel['ratio'] = (df_envel['ratio'] - 1).abs()
    df_envel[diff_name] = df_envel[diff_name].abs()
    df_envel = df_envel.groupby(['model', 'bin_mid', 'bin_width']).mean().reset_index()
    dfs_eval = []
    labels = []
    for model in df_envel['model'].unique():
        df = df_envel[df_envel['model'] == model]
        labels.append(model)
        dfs_eval.append(df)
        
    labels, dfs_eval = zip(*sorted(zip(labels, dfs_eval), key=lambda x: acc_sorted_models.get_loc(x[0])))
    labels = [MODEL_NAMING_SCHEMA[model] for model in labels]
    plot_var_dependence(dfs=dfs_eval,
                        labels=labels,
                        bin_midpoint_name='bin_mid',
                        bin_width_name='bin_width',
                        metric_names=['ratio', diff_name],
                        save_path=os.path.join(args.save_dir, 'envelope'),
                        ratio_reference_label=None,
                        xlabel=r'$p_T$ [TeV]',
                        ylabel_mapper={'ratio': r'mean $\left(\frac{\varepsilon^{-1}_{g,\mathrm{model}}}{\varepsilon^{-1}_{g,\mathrm{Pythia}}}\right)$',
                                       diff_name: r'mean $\left(\varepsilon^{-1}_{g,\mathrm{model}} - \varepsilon^{-1}_{g,\mathrm{Pythia}}\right)$'},
                        ylims=None,
                        xlog=args.xlog,
                        figsize=(7, 5),
                        h_line_position=None,
                        leg_loc='upper right',
                        # title='Envelope',
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

    # mc comparison
    load_dir = '/home/jankovys/JIDENN/logs/pythia_log_flat_70_JZ7/evaluation'
    args.save_dir = '/home/jankovys/JIDENN/logs/pythia_log_flat_70_JZ7/evaluation/compare_mc'
    args.mc_names = ['pythia', 'herwig7_dipole', 'sherpa_dire', 'herwig7',
                     'sherpa_enh_cluster_tune', 'sherpa', 'sherpa_lund', 'powheg_pythia']
    args.load_dirs = [os.path.join(load_dir, mc) for mc in args.mc_names]
    args.labels = args.mc_names
    args.model_names = ["idepart", "ipart", "depart",
                        "part", "transformer", "efn", "pfn", "fc", "highway"]
    args.metric_names = ["gluon_rejection", "quark_efficiency", 'gluon_efficiency']
    args.reference = 'pythia'
    args.ref_dir = os.path.join(load_dir, 'pythia')
    args.ylims1 = [(5, 30), (0.1, 1.), (0.1, 1.)]
    args.ylims2 = [(-0.1, 0.175), (0.7, 1.3)]
    args.xlog = True

    # pt init dist comparison
    # args.labels = ['flat', 'log flat', 'weight flat']
    # args.load_dirs = ['/home/jankovys/JIDENN/logs/pythia_flat_70_JZ7/evaluation/pythia_phys',
    #                   '/home/jankovys/JIDENN/logs/pythia_log_flat_70_JZ7/evaluation/pythia_phys_2.5TeV',
    #                   '/home/jankovys/JIDENN/logs/pythia_W_flat_70_JZ10_noL/evaluation/pythia_phys']
    # args.model_names = ["transformer", "efn", "pfn", "fc", "highway"]
    # args.save_dir = '/home/jankovys/JIDENN/logs/compare_in_pt_dist'
    # args.metric_names = ["binary_accuracy", "quark_efficiency", 'gluon_efficiency']
    # args.reference = 'flat'
    # args.ref_dir = '/home/jankovys/JIDENN/logs/pythia_flat_70_JZ7/evaluation/pythia_phys'
    # args.ylims1 = None
    # args.ylims2 = None
    # args.xlog = False

    main(args)
