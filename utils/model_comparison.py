from dataclasses import dataclass
import seaborn as sns
from typing import List, Optional, Tuple, Dict
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import numpy as np
import os
# import atlas_mpl_style as ampl
from const import MODEL_NAMING_SCHEMA, METRIC_NAMING_SCHEMA, MC_NAMING_SCHEMA
# ampl.use_atlas_style()
sns.set_theme(style="ticks")
# sns.set_context(rc={"grid.linecolor": "black"})
parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--files", nargs='*', type=str, help="csv files to load.")
parser.add_argument("--model_names", nargs='*', type=str,
                    help="names of the models.")
parser.add_argument("--x_axis_labels", nargs='*', type=str,
                    help="labels for the x axis.")
parser.add_argument("--save_dir", default=".", type=str,
                    help="Directory to save the plots to.")
parser.add_argument("--take", default=0, type=int,
                    help="Directory to save the plots to.")
parser.add_argument("--type", default="pT", type=str, help="Type of the plot.")
parser.add_argument("--compare_mc", default=False, type=bool,
                    help="Compare MC models.")


class DataSource:
    def __init__(self, path: str, model_name: str):

        self._df = pd.read_csv(path)
        self._df = self._df.drop_duplicates(subset=['cut'])
        self._df['model'] = model_name
        self.cuts = self._df['cut'].values
        self._relative_df = None

    @property
    def dataframe(self):
        return self._df

    def relative_to(self, other):
        cut1 = self.cuts
        cut2 = other.cuts
        if not np.array_equal(cut1, cut2):
            # add missing rows of cuts filled with nan
            raise ValueError("Cuts are not equal.")
            # cut1_diff = np.setdiff1d(cut1, cut2)
            # cut2_diff = np.setdiff1d(cut2, cut1)
            # if cut1_diff.size > 0:
            #     df1 = self.dataframe[self.dataframe['cut'].isin(cut1_diff)]
            #     df1['model'] = self.dataframe['model']
            #     df1 = df1.drop(columns=['cut'])
            #     other._df = pd.concat([other.dataframe, df1])

        df1 = self.dataframe.drop(columns=['cut', 'model'])
        df2 = other.dataframe.drop(columns=['cut', 'model'])
        df = df1 - df2
        df['cut'] = cut1
        df['model'] = self.dataframe['model']
        self._relative_df = df
        return df

    @property
    def relative_dataframe(self):
        if self._relative_df is None:
            raise ValueError(
                'Relative dataframe is not calculated yet. Use `relative_to` method.')
        return self._relative_df


def calculate_means(df: pd.DataFrame, metric: str = 'binary_accuracy', column_name: str = 'average relative difference') -> pd.DataFrame:
    df = df.pivot(columns=['model'], index=['cut'], values=metric)
    means = df.mean(axis=0)
    df = pd.DataFrame(means.sort_values(ascending=False))
    df.columns = [column_name]
    df = df.reset_index()
    return df


def combine_models(models: List[DataSource]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.concat([m.dataframe for m in models])
    rel_df = pd.concat([m.relative_dataframe for m in models])
    return df, rel_df


def plot_metric(df: pd.DataFrame,
                relative_df: pd.DataFrame,
                metric: str,
                x_label: str,
                save_dir: str,
                title: str,
                order: Optional[List[str]] = None,
                ylim: Optional[Tuple[float, float]] = None,):

    palette = 'coolwarm'
    fig_big = plt.figure(figsize=(16, 12))
    gs = fig_big.add_gridspec(2, hspace=0, height_ratios=[2.5, 1])
    ax1, ax2 = gs.subplots(sharex=True, sharey=False)
    sns.pointplot(x='cut', y=metric, data=df, hue='model', ci=95,
                    palette=palette, hue_order=order, ax=ax1)

    ax1.set(ylabel=METRIC_NAMING_SCHEMA[metric]
            if metric in METRIC_NAMING_SCHEMA else metric, xlabel=None)

    sns.pointplot(x='cut', y=metric, data=relative_df, hue='model', ci=95,
                    palette=palette, hue_order=order, ax=ax2, estimator=np.mean)
    ax2.set(ylabel='Relative Difference', xlabel=x_label)
    ax2.get_legend().set_visible(False)
    if ylim is not None:
        ax1.set_ylim(ylim)
    fig_big.suptitle(title, fontsize=16)
    fig_big.savefig(save_dir, bbox_inches='tight', dpi=400)
    plt.close(fig_big)


def formater(x):
    return f'{x*100:.1f}'


def bar_plot(df: pd.DataFrame,
             rel_df: pd.DataFrame,
             column_name: str,
             rel_column_name: str,
             save_dir: str,
             hue_order: Optional[List[str]] = None,
             order: Optional[List[str]] = None,) -> None:

    palette = 'Set1'
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, hspace=0, height_ratios=[2.5, 1])
    ax1, ax2 = gs.subplots(sharex=True, sharey=False)
    sns.barplot(data=df, x='DL Model', y=column_name,
                hue='model', ci=None, palette=palette, hue_order=hue_order, ax=ax1, order=order)
    for container in ax1.containers:
        ax1.bar_label(container, fmt=formater, fontsize='xx-small')
    ax1.set_ylim(0, 1.2)

    sns.barplot(data=rel_df, x='DL Model', y=rel_column_name,
                hue='model', ci=None, palette=palette, hue_order=hue_order, ax=ax2, order=order)
    for container in ax2.containers:
        ax2.bar_label(container, fmt=formater, fontsize='xx-small')
    ax2.get_legend().set_visible(False)
    ax2.set_ylim(-0.05, 0.05)
    fig.savefig(save_dir, dpi=400, bbox_inches='tight')


def dep_bar_plot(df: pd.DataFrame,
                 column_name: str,
                 save_dir: str,
                 order: Optional[List[str]] = None,
                 title: Optional[str] = None,
                 ylabel: Optional[str] = None,):

    palette = 'coolwarm'
    fig = plt.figure(figsize=(16, 6))
    # gs = fig.add_gridspec(2, hspace=0, height_ratios=[2.5, 1])
    # ax1, ax2 = gs.subplots(sharex=True, sharey=False)
    sns.pointplot(data=df, x='cut', y=column_name,
                  hue='DL Model', ci=None, palette=palette, hue_order=order)  # , ax=ax1)

    # sns.barplot(data=rel_df, x='cut', y=rel_column_name,
    #             hue='DL Model', ci=None, palette=palette, hue_order=order, ax=ax2)
    # ax2.get_legend().set_visible(False)
    plt.ylim(-0.1, 0.1)
    plt.xlabel('pT [GeV]')
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        fig.suptitle(title, fontsize=16)
    fig.savefig(save_dir, dpi=400, bbox_inches='tight')


def test(args: argparse.Namespace):
    models = ['interacting_depart', 'interacting_part', 'highway',
              'fc', 'transformer', 'part', 'depart', 'pfn', 'efn']
    # models.append('depart_100M')

    all_means = []
    all_rel_means = []
    metric_name = 'quark_efficiency'
    # metric_name = 'effective_tagging_efficiency'
    bar_name = METRIC_NAMING_SCHEMA[metric_name] if metric_name in METRIC_NAMING_SCHEMA else metric_name
    # bar_name = 'Tagging Efficiency'
    rel_bar_name = 'Difference from Pythia'
    x_label = '$p_{\mathrm{T}}$ [GeV]'
    cut_order = ['20-30', '30-40', '40-60', '60-100', '100-150', '150-200', '200-300',
                 '300-400', '400-500', '500-600', '600-800', '800-1000', '1000-1200', '1200+']

    file_string = '/home/jankovys/JIDENN/good_logs/ragged/{model}/evaluation/{mc}_50wp/pT/results.csv'.format
    save_dir = '/home/jankovys/JIDENN/good_logs/ragged/figs/mc_qeff'

    mc_models = ['pythia', 'sherpa_lund', 'herwig', 'sherpa']  # ,'herwig_dipole']
    named_mc_models = [MC_NAMING_SCHEMA[mc] for mc in mc_models]

    os.makedirs(save_dir, exist_ok=True)

    all_df = []
    all_rel_df = []
    for model in models:
        data_sources = [DataSource(file_string(model=model, mc=mc), named_mc)
                        for named_mc, mc in zip(named_mc_models, mc_models)]

        for data_source in data_sources:
            data_source.relative_to(data_sources[0])

        df, rel_df = combine_models(data_sources)
        means = calculate_means(df, metric_name, bar_name)
        rel_means = calculate_means(rel_df, metric_name, rel_bar_name)

        order = list(means['model'].values)
        plot_metric(df, rel_df, metric_name, x_label, os.path.join(save_dir, model + '.png'),
                    order=order, ylim=(0., 1.), title=MODEL_NAMING_SCHEMA[model] if model in MODEL_NAMING_SCHEMA else model)

        # means[rel_name] = - means[rel_name]
        rel_means['DL Model'] = MODEL_NAMING_SCHEMA[model] if model in MODEL_NAMING_SCHEMA else model
        means['DL Model'] = MODEL_NAMING_SCHEMA[model] if model in MODEL_NAMING_SCHEMA else model
        all_means.append(means)
        all_rel_means.append(rel_means)

        df['DL Model'] = MODEL_NAMING_SCHEMA[model] if model in MODEL_NAMING_SCHEMA else model
        rel_df['DL Model'] = MODEL_NAMING_SCHEMA[model] if model in MODEL_NAMING_SCHEMA else model
        all_df.append(df)
        all_rel_df.append(rel_df)

    # means = sorted(all_means, key=lambda x: x[bar_name].mean(), reverse=True)
    # rel_means = sorted(all_rel_means, key=lambda x: x[bar_name].mean(), reverse=True)
    means = pd.concat(all_means)
    rel_means = pd.concat(all_rel_means)
    order = rel_means[[rel_bar_name, 'DL Model']]
    order[rel_bar_name] = abs(order[rel_bar_name])
    order = order.groupby('DL Model').sum(
    ).sort_values(by=rel_bar_name, ascending=True).index.tolist()
    bar_plot(means, rel_means, bar_name, rel_bar_name, os.path.join(
        save_dir, 'bar_plot.png'), hue_order=named_mc_models, order=order)

    df = pd.concat(all_df)
    rel_df = pd.concat(all_rel_df)
    # order = rel_df[[metric_name, 'DL Model']]
    # order[metric_name] = abs(order[metric_name])
    # order = order.groupby('DL Model').sum(
    # ).sort_values(by=metric_name, ascending=True).index.tolist()
    for named_mc, mc in zip(named_mc_models, mc_models):
        dep_bar_plot(rel_df[rel_df['model'] == named_mc], metric_name, os.path.join(
            save_dir, 'diff_of_' + mc + '_from_pythia.png'), order=order, title=MC_NAMING_SCHEMA[mc] if mc in MC_NAMING_SCHEMA else mc,
            ylabel=f'{bar_name} Difference from Pythia')

    total_rel_means = rel_df[[metric_name, 'DL Model', 'cut']].groupby(['DL Model', 'cut']).sum(
    ).sort_values(by='cut', key=lambda x: pd.Categorical(x, cut_order)).reset_index()
    print(total_rel_means)
    dep_bar_plot(total_rel_means, metric_name, os.path.join(
        save_dir, 'sum_of_diffs_from_pythia.png'), order=order, title='Sum of differences from Pythia')


def dl_model_comparison(args: argparse.Namespace):
    models = ['idepart', 'ipart', 'highway', 'idepart_rel',
              'fc', 'transformer', 'part', 'depart', 'pfn', 'efn']
    # models.append('depart_100M')
    base = 'transformer'
    file_string = '/home/jankovys/JIDENN/logs/ragged_allJZ/{model}/evaluation/pythia_th/pT/results.csv'.format
    save_dir = 'logs/ragged_allJZ/figs/compare'
    os.makedirs(save_dir, exist_ok=True)
    base_source = DataSource(file_string(model=base), 'Transformer')
    data_sources = []
    for model in models:
        d_s = DataSource(file_string(model=model),
                         MODEL_NAMING_SCHEMA[model] if model in MODEL_NAMING_SCHEMA else model)
        try:
            d_s.relative_to(base_source)
        except:
            print(f'Skipping {model}')
            continue
        data_sources.append(d_s)

    df, rel_df = combine_models(data_sources)
    means = calculate_means(df, 'binary_accuracy', 'Accuracy')
    order = list(means['model'].values)
    # rel_means = calculate_means(rel_df, 'binary_accuracy', 'Difference from Transformer')
    for metric in df.columns:
        if metric == 'model' or metric == 'cut':
            continue
        ylim = (0.3, 1.0) if 'rej' not in metric and 'tag' not in metric else None
        plot_metric(df, rel_df, metric, '$p_{\mathrm{T}}$ [GeV]', save_dir +
                    f'/{metric}.png', order=order, ylim=ylim, title=metric)


def load_csvs(csvs: List[str], take: int = 0, model_names: Optional[List[str]] = None) -> List[pd.DataFrame]:
    dataframes = []
    model_names = model_names if model_names is not None else [
        f'' for i in range(len(csvs))]
    for filename, model_name in zip(csvs, model_names):
        df = pd.read_csv(filename)
        df = df.drop_duplicates(subset=['cut'])
        if take > 0:
            df = df.iloc[take:]
        df['model'] = MODEL_NAMING_SCHEMA[model_name] if model_name in MODEL_NAMING_SCHEMA else model_name
        dataframes.append(df)
    return dataframes


def calculate_relative(df: pd.DataFrame, metric, base, err_name, ordered_cut):
    df = df.pivot(columns=['model'], index=['cut'], values=metric)
    rel_err = {'model': [], err_name: []}
    for col in df.columns:
        delta = (df[col] - df[base])
        df[col] = delta
        delta = delta.mean()
        rel_err['model'].append(col)
        rel_err[err_name].append(delta)

    rel_err = pd.DataFrame(rel_err)
    rel_err = rel_err.sort_values(err_name, ascending=False)

    # sort avalues of index of a dataframe vectorized way
    def sorter(x: pd.Index) -> pd.Index:
        return

    df = df.sort_index(key=sorter)
    return rel_err, df


def main(args: argparse.Namespace, name):
    logdir = f'good_logs/ragged/{name}/evaluation/'
    model_names = ['interacting_depart', 'interacting_part', 'highway',
                   'fc', 'transformer', 'part', 'depart', 'pfn', 'efn']
    model_names = ['pythia', 'sherpa', 'sherpa_lund']
    # model_names += ['bdt']
    # used_models = ['interacting_part', 'fc', 'bdt']
    base = 'Transformer'
    base = 'pythia'
    # save_dir = logdir + 'figs/' + str(args.take) + '/'

    save_dir = logdir + 'figs/'

    if args.type == 'pT' or args.type == 'pT_no_cut':
        eval_dir = f'{args.type}'
        ylabel = '$p_{\mathrm{T}}$ [GeV]'
        ylabel_averaged = '$p_{\mathrm{T}}$ Averaged'
        cut = ['20-30', '30-40', '40-60', '60-100', '100-150', '150-200', '200-300',
               '300-400', '400-500', '500-600', '600-800', '800-1000', '1000-1200', '1200+']
        save_dir += f'{args.type}_{args.take}/'
    elif args.type == 'eta':
        eval_dir = 'eval_eta'
        cut = ["0.0-0.1", "0.1-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", "0.9-1.1",
               "1.1-1.3", "1.3-1.5", "1.5-1.7", "1.7-1.9", "1.9-2.1", "2.1+"]
        ylabel = '$|\eta|$'
        ylabel_averaged = '$|\eta|$ Averaged'
        save_dir += f'eta_{args.take}/'
    elif args.type == 'pileup':
        eval_dir = 'eval_pileup'
        cut = ["0-20", "20-25", "25-30", "30-35",
               "35-40", "40-50", "50-55", "55-60", "60+"]
        ylabel = r'$\mu$'
        ylabel_averaged = r'$\mu$ Averaged'
        save_dir += f'pileup_{args.take}/'
    elif args.type == 'JZ' or args.type == 'JZ_full_cut' or args.type == 'JZ_mid_cut':
        eval_dir = f'eval_{args.type}'
        cut = ["1", "2", "3", "4", "5", ]
        ylabel = 'JZ'
        ylabel_averaged = 'JZ Averaged'
        save_dir += f'{args.type}_{args.take}/'
    else:
        raise ValueError('Unknown type')


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    if args.compare_mc:
        test(args)
    else:
        dl_model_comparison(args)
