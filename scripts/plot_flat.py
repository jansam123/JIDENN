import os
import sys
sys.path.append(os.getcwd())
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import atlasify
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--load_path", type=str, help="Path to the root file")


def sns_label_plotting(df, save_path, bins=None):
    pt_name = r"$p_{\mathrm{T}}$ [TeV]"
    label_name = "label"
    if bins is None:
        bins = list(np.linspace(0.06, 4.6, 71))
    df = df.rename(columns={"jets_pt": pt_name, "jets_PartonTruthLabelID": label_name})
    ax = sns.histplot(x=df[pt_name].values, hue=df[label_name].values, weights=df["weight_spectrum"].values, multiple="layer", log_scale=(
        False, True), bins=bins, common_norm=False, stat="count", palette='Set1', element="step", fill=False, hue_order=['quark', 'gluon'])

    lss = ['-', '--']
    # handles = ax.legend_.legendHandles[::-1]
    # for line, ls, handle in zip(ax.lines, lss, handles):
    #     line.set_linestyle(ls)
    #     handle.set_ls(ls)

    # ax.legend_.set_title(None)
    plt.xlabel(pt_name, horizontalalignment='right', x=1.0)
    plt.ylabel(r"Counts", horizontalalignment='right', y=1.0)
    plt.ylim(1e3, 1e6)
    plt.xlim(0.2, 2.5)
    atlasify.atlasify(
        axes=ax,
        subtext='13 TeV, Pythia8\n' + r'anti-$k_{\mathrm{T}}$, $R = 0.4$ PFlow jets',
        atlas='Simulation Internal',
        font_size=15,
        sub_font_size=15,
        label_font_size=15,
    )
    plt.savefig(f"{save_path}/flat_pt.pdf", bbox_inches='tight')
    plt.savefig(f"{save_path}/flat_pt.png", dpi=400, bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close()


def sns_label_plotting_noW(df, save_path, bins=None):
    pt_name = r"$p_{\mathrm{T}}$ [TeV]"
    label_name = "label"
    if bins is None:
        bins = list(np.linspace(0.06, 4.6, 71))
    df = df.rename(columns={"jets_pt": pt_name, "jets_PartonTruthLabelID": label_name})
    ax = sns.histplot(data=df, x=pt_name, hue=label_name, multiple="layer", log_scale=(
        False, True), bins=bins, common_norm=False, stat="count", palette='Set1', element="step", fill=False, hue_order=['quark', 'gluon'])
    lss = ['-', '--']
    # handles = ax.legend_.legendHandles[::-1]
    # for line, ls, handle in zip(ax.lines, lss, handles):
    #     line.set_linestyle(ls)
    #     handle.set_ls(ls)
    # ax.legend_.set_title(None)
    plt.xlabel(pt_name, horizontalalignment='right', x=1.0)
    plt.ylabel(r"Counts", horizontalalignment='right', y=1.0)
    plt.ylim(1e3, 1e6)
    plt.xlim(0.2, 2.5)
    atlasify.atlasify(
        axes=ax,
        subtext='13 TeV, Pythia8\n' + r'anti-$k_{\mathrm{T}}$, $R = 0.4$ PFlow jets',
        atlas='Simulation Internal',
        font_size=14,
        sub_font_size=14,
        label_font_size=14,
    )
    plt.savefig(f"{save_path}/flat_pt_noW.pdf", bbox_inches='tight')
    plt.savefig(f"{save_path}/flat_pt_noW.png", dpi=400, bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close()


def load_dataframe(path, vars):
    from jidenn.data.JIDENNDataset import JIDENNDataset
    ds = JIDENNDataset.load(path)
    ds = ds.remap_data(lambda x: {var: x[var] for var in vars}).to_pandas()
    return ds


if __name__ == "__main__":
    args  = parser.parse_args()
    HUE_MAPPER = {1: 'quark', 2: 'quark', 3: 'quark', 4: 'quark', 5: 'quark', 6: 'quark', 21: 'gluon'}
    train_path = args.load_path
    try:
        df = pd.read_csv(os.path.join(train_path, "dataset.csv"))
    except FileNotFoundError:
        df = load_dataframe(train_path, ["jets_pt", "jets_PartonTruthLabelID", "weight_spectrum"])
        df.to_csv(os.path.join(train_path, "dataset.csv"))

    df['jets_pt'] *= 1e-6
    bins = list(df.query('jets_PartonTruthLabelID == 21')[['jets_pt', 'weight_spectrum']].groupby('weight_spectrum').min()['jets_pt'].values + 1e-6)
    bins.append(df.query('jets_PartonTruthLabelID != 21')[['jets_pt', 'weight_spectrum']].groupby('weight_spectrum').max()['jets_pt'].values[-1] + 1e-6)
    df['jets_PartonTruthLabelID'] = df['jets_PartonTruthLabelID'].replace(HUE_MAPPER)
    bins = np.array(bins)
    bins = bins[np.argsort(bins)]
    _, indicies = np.unique(np.round(bins, 3), return_index=True)
    bins = bins[indicies]
    print(bins)
    sns_label_plotting(df, train_path, bins=bins)
    sns_label_plotting_noW(df, train_path, bins=bins)
    print('DONE')
