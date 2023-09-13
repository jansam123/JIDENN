import seaborn as sns
import matplotlib.pyplot as plt
from puma import Histogram, HistogramPlot
import pandas as pd
import numpy as np
import os
import atlasify



def sns_label_plotting(df, save_path):
    pt_name = r"$p_{\mathrm{T}}$ [TeV]"
    label_name = "label"
    bins = list(np.linspace(0.06, 2.5, 101))
    df = df.rename(columns={"jets_pt": pt_name, "jets_PartonTruthLabelID": label_name})
    ax = sns.histplot(data=df, x=pt_name, hue=label_name, weights="weight_spectrum", multiple="layer", log_scale=(
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
    plt.xlim(0.06, 2.5)
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


def sns_label_plotting_noW(df, save_path):
    pt_name = r"$p_{\mathrm{T}}$ [TeV]"
    label_name = "label"
    bins = list(np.linspace(0.06, 2.5, 101))
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
    plt.xlim(0.06, 2.5)
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
    plt.ylim(4e-2, 1e11)
    plt.xlim(0.04, 2.5)
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
    train_path = "/home/jankovys/JIDENN/data/pythia_nW_flat_100_JZ7/train"
    try:
        df = pd.read_csv(os.path.join(train_path, "dataset.csv"))
    except FileNotFoundError:
        df = load_dataframe(train_path, ["jets_pt", "jets_PartonTruthLabelID", "weight_spectrum"])
        df.to_csv(os.path.join(train_path, "dataset.csv"))

    df['jets_pt'] *= 1e-6
    df = df[df['jets_pt'] > 0.06]
    df = df[df['jets_pt'] < 2.6]
    # remap the labels
    df['jets_PartonTruthLabelID'] = df['jets_PartonTruthLabelID'].replace(HUE_MAPPER)
    sns_label_plotting(df, train_path)
    sns_label_plotting_noW(df, train_path)

    phys_path = "/home/jankovys/JIDENN/data/altMC_phys/Pythia8EvtGen_A14NNPDF23LO_jetjet"
    try:
        df = pd.read_csv(os.path.join(phys_path, "pythia_physical.csv"))
    except FileNotFoundError:
        df = load_dataframe(phys_path, ["jets_pt", "jets_PartonTruthLabelID", "weight"])
        df.to_csv(os.path.join(phys_path, "pythia_physical.csv"))
    df['jets_pt'] *= 1e-6
    df['jets_PartonTruthLabelID'] = df['jets_PartonTruthLabelID'].replace(HUE_MAPPER)
    sns_label_plotting_phys(df, phys_path)
