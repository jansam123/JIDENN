import seaborn as sns
import matplotlib.pyplot as plt
from puma import Histogram, HistogramPlot
import pandas as pd
import numpy as np
import atlasify

from jidenn.data.JIDENNDataset import JIDENNDataset

def sns_label_plotting(df):
    pt_name = r"$p_{\mathrm{T}}$ [TeV]"
    label_name = "label"
    bins = list(np.linspace(0.06, 4.6, 70))
    df = df.rename(columns={"jets_pt": pt_name, "jets_PartonTruthLabelID": label_name})
    ax = sns.histplot(data=df, x=pt_name, hue=label_name, weights="weight_spectrum", multiple="layer", log_scale=(
        False, True), bins=bins, common_norm=False, stat="count", palette='Set1', element="step", fill=False, hue_order=['quark', 'gluon'])
    
    lss = ['-', '--']
    handles = ax.legend_.legendHandles[::-1]
    for line, ls, handle in zip(ax.lines, lss, handles):
        line.set_linestyle(ls)
        handle.set_ls(ls)
        
    ax.legend_.set_title(None)
    plt.xlabel(r"Jet $p_{\mathrm{T}}$ [TeV]", horizontalalignment='right', x=1.0)
    plt.ylabel(r"Counts", horizontalalignment='right', y=1.0)
    plt.ylim(1e0, 5e6)
    plt.xlim(0.06, 4.6)
    atlasify.atlasify(
        axes=ax,
        subtext='Simulation Preliminary \n 13 TeV',
        sub_font_size=10,
    )
    plt.savefig("data/pythia_W_flat_70_JZ10/train/flat_pt.pdf", bbox_inches='tight')
    plt.savefig("data/pythia_W_flat_70_JZ10/train/flat_pt.png", dpi=400, bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close()


def sns_label_plotting_noW(df):
    pt_name = r"$p_{\mathrm{T}}$ [TeV]"
    label_name = "label"
    bins = list(np.linspace(0.06, 4.6, 70))
    df = df.rename(columns={"jets_pt": pt_name, "jets_PartonTruthLabelID": label_name})
    ax = sns.histplot(data=df, x=pt_name, hue=label_name, multiple="layer", log_scale=(
        False, True), bins=bins, common_norm=False, stat="count", palette='Set1', element="step", fill=False, hue_order=['quark', 'gluon'])
    lss = ['-', '--']
    handles = ax.legend_.legendHandles[::-1]
    for line, ls, handle in zip(ax.lines, lss, handles):
        line.set_linestyle(ls)
        handle.set_ls(ls)
    ax.legend_.set_title(None)
    plt.xlabel(r"Jet $p_{\mathrm{T}}$ [TeV]", horizontalalignment='right', x=1.0)
    plt.ylabel(r"Counts", horizontalalignment='right', y=1.0)
    plt.ylim(6e2, 2e6)
    plt.xlim(0.06, 4.6)
    atlasify.atlasify(
        axes=ax,
        subtext='Simulation Preliminary \n 13 TeV',
        sub_font_size=10,
    )
    plt.savefig("data/pythia_W_flat_70_JZ10/train/flat_pt_noW.pdf", bbox_inches='tight')
    plt.savefig("data/pythia_W_flat_70_JZ10/train/flat_pt_noW.png", dpi=400, bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close()


def sns_label_plotting_phys(df):
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
        
    plt.xlabel(r"Jet $p_{\mathrm{T}}$ [TeV]", horizontalalignment='right', x=1.0)
    plt.ylabel(r"Counts", horizontalalignment='right', y=1.0)
    plt.ylim(1e-6, 6e12)
    plt.xlim(0.04, 5.)
    atlasify.atlasify(
        axes=ax,
        subtext='Simulation Preliminary \n 13 TeV',
        sub_font_size=10,
    )
    plt.savefig("data/pythia_physical/phys_pt.pdf", bbox_inches='tight')
    plt.savefig("data/pythia_physical/phys_pt.png", dpi=400, bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close()

def load_dataframe(path):
    ds = JIDENNDataset.load(path)
    ds = ds.map(lambda x: {'jets_pt':x['jets_pt'], 'weight_spectrum':x['weight_spectrum'], 'jets_PartonTruthLabelID':x['jets_PartonTruthLabelID']}).to_pandas()
    return ds

if __name__ == "__main__":
    HUE_MAPPER = {1: 'quark', 2: 'quark', 3: 'quark', 4: 'quark', 5: 'quark', 6: 'quark', 21: 'gluon'}
    try:
        df = pd.read_csv("data/pythia_W_flat_70_JZ10/train/dataset.csv")
    except FileNotFoundError:
        df = load_dataframe("data/pythia_W_flat_70_JZ10/train")
        df.to_csv("data/pythia_W_flat_70_JZ10/train/dataset.csv")
    df['jets_pt'] *= 1e-6
    df['weight_spectrum'] /= 1e4
    df = df[df['jets_pt'] > 0.07]
    df = df[df['jets_pt'] < 5]
    # remap the labels
    df['jets_PartonTruthLabelID'] = df['jets_PartonTruthLabelID'].replace(HUE_MAPPER)
    sns_label_plotting(df)
    sns_label_plotting_noW(df)

    df = pd.read_csv("data/pythia_physical/pythia_physical.csv")
    df['jets_pt'] *= 1e-6
    df['jets_PartonTruthLabelID'] = df['jets_PartonTruthLabelID'].replace(HUE_MAPPER)
    sns_label_plotting_phys(df)
