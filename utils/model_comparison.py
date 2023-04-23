import seaborn as sns
from typing import List
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import numpy as np
import os
# import atlas_mpl_style as ampl
from const import MODEL_NAMING_SCHEMA, MATRIC_NAMING_SCHEMA
# ampl.use_atlas_style()
sns.set_theme(style="ticks")
# sns.set_context(rc={"grid.linecolor": "black"})
parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--files", nargs='*', type=str, help="csv files to load.")
parser.add_argument("--model_names", nargs='*', type=str, help="names of the models.")
parser.add_argument("--x_axis_labels", nargs='*', type=str, help="labels for the x axis.")
parser.add_argument("--save_dir", default=".", type=str, help="Directory to save the plots to.")
parser.add_argument("--take", default=0, type=int, help="Directory to save the plots to.")
parser.add_argument("--type", default="pT", type=str, help="Type of the plot.")


def main(args: argparse.Namespace):
    logdir = 'good_logs/comparison_12e/'
    model_names = ['interacting_depart', 'interacting_depart_no_norm', 'interacting_part', 'highway',
                   'fc', 'transformer', 'part', 'depart', 'pfn', 'efn', 'depart_rel']
    # model_names += ['bdt']
    # used_models = ['interacting_part', 'fc', 'bdt']
    base = 'Transformer'
    # save_dir = logdir + 'figs/' + str(args.take) + '/'

    save_dir = logdir + 'figs/'

    if args.type == 'pT' or args.type == 'pT_no_cut':
        eval_dir = f'eval_{args.type}'
        ylabel = '$p_{\mathrm{T}}$ [GeV]'
        ylabel_averaged = '$p_{\mathrm{T}}$ Averaged Accuracy Difference'
        cut = ['20-30', '30-40', '40-60', '60-100', '100-150', '150-200', '200-300',
               '300-400', '400-500', '500-600', '600-800', '800-1000', '1000-1200', '1200+']
        save_dir += f'{args.type}_{args.take}/'
    elif args.type == 'eta':
        eval_dir = 'eval_eta'
        cut = ["0.0-0.1", "0.1-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", "0.9-1.1",
               "1.1-1.3", "1.3-1.5", "1.5-1.7", "1.7-1.9", "1.9-2.1", "2.1+"]
        ylabel = '$|\eta|$'
        ylabel_averaged = '$|\eta|$ Averaged Accuracy Difference'
        save_dir += f'eta_{args.take}/'
    elif args.type == 'pileup':
        eval_dir = 'eval_pileup'
        cut = ["0-20", "20-25", "25-30", "30-35", "35-40", "40-50", "50-55", "55-60", "60+"]
        ylabel = r'$\mu$'
        ylabel_averaged = r'$\mu$ Averaged Accuracy Difference'
        save_dir += f'pileup_{args.take}/'
    elif args.type == 'JZ' or args.type == 'JZ_full_cut' or args.type == 'JZ_mid_cut':
        eval_dir = f'eval_{args.type}'
        cut = ["1", "2", "3", "4", "5", ]
        ylabel = 'JZ'
        ylabel_averaged = 'JZ Averaged Accuracy Difference'
        save_dir += f'{args.type}_{args.take}/'
    else:
        raise ValueError('Unknown type')

    palette = 'coolwarm'  # sns.diverging_palette(250, 30, l=65, center="dark",n=len(model_names))
    cmap = 'coolwarm'  # sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)

    result_csvs = []
    roc_csvs = []
    usable_models = []
    for model in model_names:
        csv_file = logdir + model + f'/{eval_dir}/results.csv'
        roc_file = logdir + model + f'/{eval_dir}/base/figs/csv/roc.csv'
        if os.path.isfile(csv_file):
            result_csvs.append(csv_file)
            usable_models.append(model)
            roc_csvs.append(roc_file)

    os.makedirs(save_dir, exist_ok=True)
    take = args.take
    print(f'Using {usable_models} models')

    # load the CSV files into a list of pandas DataFrames
    dataframes = []
    roc_dataframes = []
    for filename, model_name, roc_filename in zip(result_csvs, usable_models, roc_csvs):
        df = pd.read_csv(filename)
        df = df.drop_duplicates(subset=['cut'])
        try:
            roc_df = pd.read_csv(roc_filename)
            roc_df['model'] = model_name
            roc_df = roc_df.iloc[::100, :]
        except FileNotFoundError:
            roc_df = pd.DataFrame()
            print(f'No ROC file for {model_name}')
        df['model'] = MODEL_NAMING_SCHEMA[model_name] if model_name in MODEL_NAMING_SCHEMA else model_name
        # roc_df['model'] = roc_df['model'] + ' (AUC = ' + auc(roc_df['FPR'], roc_df['TPR']).round(3).astype(str) + ')'
        if take > 0:
            df = df.iloc[take:]
        dataframes.append(df)
        roc_dataframes.append(roc_df)

    # concatenate the DataFrames into a single DataFrame
    df = pd.concat(dataframes)
    roc_df = pd.concat(roc_dataframes, ignore_index=True)
    df.to_csv(save_dir + 'all_data.csv')

    os.makedirs(save_dir, exist_ok=True)
    cut = cut[args.take:]

    new_df = df[['cut', 'binary_accuracy', 'model']]
    new_df = new_df.pivot(columns=['model'])
    new_df = new_df.loc[:, 'binary_accuracy']
    print(new_df)
    err_name = ylabel_averaged
    rel_err = {'model': [], err_name: []}
    for col in new_df.columns:
        delta = (new_df[col] - new_df[base])
        new_df[col] = delta
        delta = delta.mean()
        rel_err['model'].append(col)
        rel_err[err_name].append(delta)

    rel_err = pd.DataFrame(rel_err)
    rel_err = rel_err.sort_values(err_name, ascending=False)

    # rel_err = rel_err[rel_err['model'].isin(used_models)]
    # df = df[df['model'].isin(used_models)]
    # roc_df = roc_df[roc_df['model'].isin(used_models)]

    for metric in df.columns.drop(['model', 'cut']):
        fig = plt.figure(figsize=(15, 9))
        ax = sns.pointplot(x='cut', y=metric, data=df, hue='model', ci=95,
                           palette=palette, hue_order=rel_err['model'])

        plt.xlabel(ylabel)
        plt.ylabel(MATRIC_NAMING_SCHEMA[metric] if metric in MATRIC_NAMING_SCHEMA else metric)
        plt.savefig(save_dir + f'{metric}.jpg', dpi=300, bbox_inches='tight')
        plt.close()

    fig_big = plt.figure(figsize=(16, 12))
    gs = fig_big.add_gridspec(2, hspace=0, height_ratios=[2.5, 1])
    ax1, ax2 = gs.subplots(sharex=True, sharey=False)
    sns.pointplot(x='cut', y='binary_accuracy', data=df, hue='model', ci=95,
                  palette=palette, hue_order=rel_err['model'], ax=ax1)
    ax1.set(ylabel='Accuracy', xlabel=None)
    if args.take == 0:
        ax1.set_ylim(0.6, 0.83)

    fig, ax = plt.subplots(figsize=(14, 9))
    max_delta = abs(rel_err[err_name]).max()
    sns.barplot(data=rel_err, x=err_name, y='model', palette=palette, dodge=False, ax=ax)
    ax.set_xlim(-max_delta, max_delta)
    fig.savefig(save_dir + f'relative_error.jpg', dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(new_df)
    new_df = new_df.set_index(pd.Index(cut))
    new_df = new_df.reindex(rel_err['model'], axis=1)

    fig, ax = plt.subplots(figsize=(14, 14))
    sns.heatmap(new_df, annot=True, fmt='.3f', cmap=cmap + '_r', cbar=False, ax=ax)
    ax.set_xlabel('Model')
    ax.set_ylabel(ylabel)
    fig.savefig(save_dir + f'heatmap.jpg', dpi=300, bbox_inches='tight')
    plt.close(fig)

    new_df['metric'] = 'Relative Accuracy'
    new_df = new_df.reset_index()
    new_df = new_df.melt(id_vars=['index', 'metric'], var_name='model', value_name='value')

    sns.pointplot(data=new_df, x='index', y='value', hue='model', ci=95, palette=palette, ax=ax2, legend=False)
    for ax in fig.get_axes():
        ax.label_outer()
    ax2.set(xlabel=ylabel, ylabel=f'Difference from {base}')
    ax2.get_legend().set_visible(False)
    if args.take == 0:
        ax2.set_ylim(-0.030, 0.014)
    fig_big.savefig(save_dir + f'relative_accuracy.jpg', dpi=200, bbox_inches='tight')
    plt.close(fig_big)

    roc_df['FPR'] = roc_df['FPR'] * 100
    roc_df['TPR'] = roc_df['TPR'] * 100
    fig = plt.figure(figsize=(8, 8))
    sns.lineplot(data=roc_df, x='FPR', y='TPR', hue='model', palette=palette, linewidth=2)
    sns.lineplot(x=[0, 50, 100], y=[0, 50, 100], label=f'Random',
                 linewidth=1, linestyle='--', color='darkred', alpha=0.5)
    plt.plot([0, 0, 100], [0, 100, 100], color='darkgreen', linestyle='-.', label='Ideal', alpha=0.5)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.savefig(save_dir + f'roc.jpg', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
