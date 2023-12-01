import seaborn as sns
import matplotlib.pyplot as plt
from puma import Histogram, HistogramPlot
import pandas as pd
import numpy as np
import os
import atlasify
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--load_path", type=str, help="Path to the root file")


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
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    
    plt.setp(ax.get_legend().get_texts(), fontsize='15')
    plt.xlabel(pt_name, horizontalalignment='right', x=1.0, fontsize=15)
    plt.ylabel(r"Counts", horizontalalignment='right', y=1.0, fontsize=15)
    plt.ylim(1e-1, 6e9)
    plt.xlim(0.2, 2.5)
    atlasify.atlasify(
        axes=ax,
        subtext='13 TeV, Pythia8\n' + r'anti-$k_{\mathrm{T}}$, $R = 0.4$ PFlow jets',
        atlas='Simulation Internal',
        font_size=15,
        sub_font_size=15,
        label_font_size=15,
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
    args = parser.parse_args()
    phys_path = args.load_path
    try:
        df = pd.read_csv(os.path.join(phys_path, "pythia_physical.csv"))
    except FileNotFoundError:
        df = load_dataframe(phys_path, ["jets_pt", "jets_PartonTruthLabelID", "weight"])
        df.to_csv(os.path.join(phys_path, "pythia_physical.csv"))
    df['jets_pt'] *= 1e-6
    df['jets_PartonTruthLabelID'] = df['jets_PartonTruthLabelID'].replace(HUE_MAPPER)
    sns_label_plotting_phys(df, phys_path)
    print('Done!')
