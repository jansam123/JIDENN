import seaborn as sns
from typing import List
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import numpy as np
import os
# import atlas_mpl_style as ampl
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


def main(args: argparse.Namespace):
    logdir = 'good_logs/comparison_small/'
    eval_dir = 'eval'

    ylabel = 'corrected_averageInteractionsPerCrossing[0]'
    ylabel_averaged = 'corrected_averageInteractionsPerCrossing[0] Averaged Accuracy Difference'

    ylabel = '$|\eta|$'
    ylabel_averaged = '$|\eta|$ Averaged Accuracy Difference'
    
    ylabel = '$p_{\mathrm{T}}$ [GeV]'
    ylabel_averaged = '$p_{\mathrm{T}}$ Averaged Accuracy Difference'

    model_names = ['interacting_depart', 'interacting_part', 'highway',
                   'basic_fc', 'transformer', 'part', 'depart', 'interacting_depart_40M', 'pfn']
    # model_names += ['bdt']
    # used_models = ['interacting_part', 'basic_fc', 'bdt']
    used_models = model_names
    base = 'transformer'
    # save_dir = logdir + 'figs/' + str(args.take) + '/'

    save_dir = logdir + 'figs/' + 'pT_' + str(args.take) + '/'

    cut = ["0-20", "20-25", "25-30", "30-35", "35-40", "40-50", "50-55", "55-60", "60+"]


    cut = ["0.0-0.1", "0.1-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", "0.9-1.1",
           "1.1-1.3", "1.3-1.5", "1.5-1.7", "1.7-1.9", "1.9-2.1", "2.1+"]
    
    cut = ['20-30', '30-40', '40-60', '60-100', '100-150', '150-200', '200-300',
           '300-400', '400-500', '500-600', '600-800', '800-1000', '1000-1200', '1200+']

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

    # load the CSV files into a list of pandas DataFrames
    dataframes = []
    roc_dataframes = []
    for filename, model_name, roc_filename in zip(result_csvs, usable_models, roc_csvs):
        df = pd.read_csv(filename)
        try:
            roc_df = pd.read_csv(roc_filename)
            roc_df['model'] = model_name
            roc_df = roc_df.iloc[::100, :]
        except FileNotFoundError:
            roc_df = pd.DataFrame()
            print(f'No ROC file for {model_name}')
        df['model'] = model_name
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

    rel_err = rel_err[rel_err['model'].isin(used_models)]
    df = df[df['model'].isin(used_models)]
    roc_df = roc_df[roc_df['model'].isin(used_models)]

    for metric in df.columns.drop(['model', 'cut']):
        fig = plt.figure(figsize=(15, 9))
        ax = sns.pointplot(x='cut', y=metric, data=df, hue='model', ci=95,
                           palette=palette, hue_order=rel_err['model'])

        plt.xlabel(ylabel)
        plt.ylabel(metric)
        plt.savefig(save_dir + f'{metric}.jpg', dpi=300)
        plt.close()

    fig = plt.figure(figsize=(14, 9))
    max_delta = abs(rel_err[err_name]).max()
    p = sns.barplot(data=rel_err, x=err_name, y='model', palette=palette, dodge=False)
    p.axes.set_xlim(-max_delta, max_delta)
    plt.savefig(save_dir + f'relative_error.jpg', dpi=300)
    plt.close()

    new_df = new_df.set_index(pd.Index(cut))
    new_df = new_df.reindex(rel_err['model'], axis=1)

    fig = plt.figure(figsize=(14, 14))
    sns.heatmap(new_df, annot=True, fmt='.3f', cmap=cmap + '_r', cbar=False)
    plt.xlabel('Model')
    plt.ylabel(ylabel)
    plt.savefig(save_dir + f'heatmap.jpg', dpi=300)
    plt.close()

    new_df['metric'] = 'Relative Accuracy'
    new_df = new_df.reset_index()
    new_df = new_df.melt(id_vars=['index', 'metric'], var_name='model', value_name='value')

    fig = plt.figure(figsize=(15, 7))
    sns.pointplot(data=new_df, x='index', y='value', hue='model', ci=95, palette=palette)
    plt.xlabel(ylabel)
    plt.ylabel('Relative Accuracy')
    plt.savefig(save_dir + f'relative_accuracy.jpg', dpi=300)
    plt.close()

    roc_df['FPR'] = roc_df['FPR'] * 100
    roc_df['TPR'] = roc_df['TPR'] * 100
    fig = plt.figure(figsize=(8, 8))
    sns.lineplot(data=roc_df, x='FPR', y='TPR', hue='model', palette=palette,
                 linewidth=2)  # , hue_order=rel_err['model'])
    sns.lineplot(x=[0, 50, 100], y=[0, 50, 100], label=f'Random',
                 linewidth=1, linestyle='--', color='darkred', alpha=0.5)
    plt.plot([0, 0, 100], [0, 100, 100], color='darkgreen', linestyle='-.', label='Ideal', alpha=0.5)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.savefig(save_dir + f'roc.jpg', dpi=300)
    plt.close()


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
