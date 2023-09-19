import seaborn as sns
import matplotlib.pyplot as plt
from puma import Histogram, HistogramPlot
import pandas as pd
import numpy as np
import os
import atlasify


def sns_label_plotting(df, save_path, bins=None):
    pt_name = r"$p_{\mathrm{T}}$ [TeV]"
    label_name = "label"
    if bins is None:
        bins = list(np.linspace(0.06, 4.6, 71))
    df = df.rename(columns={"jets_pt": pt_name, "jets_PartonTruthLabelID": label_name})
    ax = sns.histplot(x=df[pt_name].values, hue=df[label_name].values, weights=df["weight_spectrum"].values, multiple="layer", log_scale=(
        False, True), bins=bins, common_norm=False, stat="count", palette='Set1', element="step", fill=False, hue_order=['quark', 'gluon'])

    lss = ['-', '--']
    handles = ax.legend_.legendHandles[::-1]
    for line, ls, handle in zip(ax.lines, lss, handles):
        line.set_linestyle(ls)
        handle.set_ls(ls)

    ax.legend_.set_title(None)
    plt.xlabel(pt_name, horizontalalignment='right', x=1.0)
    plt.ylabel(r"Counts", horizontalalignment='right', y=1.0)
    plt.ylim(1e3, 1e6)
    plt.xlim(0.06, 4.6)
    atlasify.atlasify(
        axes=ax,
        subtext='13 TeV',
        atlas='Simulation Internal',
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
    handles = ax.legend_.legendHandles[::-1]
    for line, ls, handle in zip(ax.lines, lss, handles):
        line.set_linestyle(ls)
        handle.set_ls(ls)
    ax.legend_.set_title(None)
    plt.xlabel(pt_name, horizontalalignment='right', x=1.0)
    plt.ylabel(r"Counts", horizontalalignment='right', y=1.0)
    plt.ylim(1e3, 1e6)
    plt.xlim(0.06, 4.6)
    atlasify.atlasify(
        axes=ax,
        subtext='13 TeV',
        atlas='Simulation Internal',
    )
    plt.savefig(f"{save_path}/flat_pt_noW.pdf", bbox_inches='tight')
    plt.savefig(f"{save_path}/flat_pt_noW.png", dpi=400, bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close()


def sns_label_plotting_phys(df, save_path):
    pt_name = r"$p_{\mathrm{T}}$ [TeV]"
    label_name = "label"
    df = df.rename(columns={"jets_pt": pt_name, "jets_PartonTruthLabelID": label_name})
    ax = sns.histplot(data=df, x=pt_name, hue=label_name, weights="weight", multiple="layer", log_scale=(
        False, True), bins=100, common_norm=False, stat="count", palette='Set1', element="step", fill=False, hue_order=['quark', 'gluon'])
    ax.legend_.set_title(None)

    lss = ['-', '--']
    handles = ax.legend_.legendHandles[::-1]
    for line, ls, handle in zip(ax.lines, lss, handles):
        line.set_linestyle(ls)
        handle.set_ls(ls)

    plt.xlabel(pt_name, horizontalalignment='right', x=1.0)
    plt.ylabel(r"Counts", horizontalalignment='right', y=1.0)
    plt.ylim(4e-2, 1e9)
    plt.xlim(0.16, 2.5)
    atlasify.atlasify(
        axes=ax,
        subtext='13 TeV',
        atlas='Simulation Internal',
    )
    plt.savefig(f"{save_path}/phys_pt.pdf", bbox_inches='tight')
    plt.savefig(f"{save_path}/phys_pt.png", dpi=400, bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close()


def load_dataframe(path, vars):
    from jidenn.data.JIDENNDataset import JIDENNDataset
    ds = JIDENNDataset.load(path)
    ds = ds.remap_data(lambda x: {var: x[var] for var in vars}).to_pandas()
    return ds


if __name__ == "__main__":
    HUE_MAPPER = {1: 'quark', 2: 'quark', 3: 'quark', 4: 'quark', 5: 'quark', 6: 'quark', 21: 'gluon'}
    train_path = "/home/jankovys/JIDENN/data/pythia_W_flat_70_JZ10_noL/train"
    try:
        df = pd.read_csv(os.path.join(train_path, "dataset.csv"))
    except FileNotFoundError:
        df = load_dataframe(train_path, ["jets_pt", "jets_PartonTruthLabelID", "weight_spectrum"])
        df.to_csv(os.path.join(train_path, "dataset.csv"))

    df['jets_pt'] *= 1e-6
    # print(df['jets_pt'].min(), df['jets_pt'].max())
    # print(df.query('jets_PartonTruthLabelID == 21')['weight_spectrum'].sort_values().unique())
    bins = list(df.query('jets_PartonTruthLabelID == 21')[['jets_pt', 'weight_spectrum']].groupby('weight_spectrum').min()['jets_pt'].values + 1e-6)
    bins.append(df.query('jets_PartonTruthLabelID != 21')[['jets_pt', 'weight_spectrum']].groupby('weight_spectrum').max()['jets_pt'].values[-1] + 1e-6)
    # print(df.query('jets_PartonTruthLabelID != 21')['weight_spectrum'].sort_values().unique())
    # print(df.query('jets_PartonTruthLabelID != 21')[['jets_pt', 'weight_spectrum']].groupby('weight_spectrum').min()['jets_pt'].values)
    # print(df.query('jets_PartonTruthLabelID != 21')[['jets_pt', 'weight_spectrum']].groupby('weight_spectrum').max()['jets_pt'].values)
    df['jets_PartonTruthLabelID'] = df['jets_PartonTruthLabelID'].replace(HUE_MAPPER)
    df['weight_spectrum'] *= 1e-4
    # remap the labels
    bins = np.array(bins) 
    bins = bins[np.argsort(bins)]
    _, indicies = np.unique(np.round(bins, 3), return_index=True)
    bins = bins[indicies]
    print(bins) 
    sns_label_plotting(df, train_path, bins=bins)
    sns_label_plotting_noW(df, train_path, bins=bins)

    phys_path = "/home/jankovys/JIDENN/data/altMC_phys_JZ3/Pythia8EvtGen_A14NNPDF23LO_jetjet"
    phys_path_dir = "/home/jankovys/JIDENN/data/altMC_phys_JZ3/"
    for file in os.listdir(phys_path_dir):
        phys_path = os.path.join(phys_path_dir, file)
        try:
            try:
                df = pd.read_csv(os.path.join(phys_path, "pythia_physical.csv"))
            except FileNotFoundError:
                df = load_dataframe(phys_path, ["jets_pt", "jets_PartonTruthLabelID", "weight"])
                df.to_csv(os.path.join(phys_path, "pythia_physical.csv"))
            df['jets_pt'] *= 1e-6
            df = df[df['jets_pt'] > 0.16]
            df = df[df['jets_pt'] < 2.5]
            df['jets_PartonTruthLabelID'] = df['jets_PartonTruthLabelID'].replace(HUE_MAPPER)
            sns_label_plotting_phys(df, phys_path)
        except:
            print(f"Failed to plot {file}")
            continue
