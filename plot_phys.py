import seaborn as sns
import matplotlib.pyplot as plt
from puma import Histogram, HistogramPlot
import pandas as pd
import atlasify


def puma_plotting(df):
    hists = [Histogram(df[df['JZ_slice'] == i]['jets_pt'].to_numpy(), weights=df[df['JZ_slice'] == i]['weight'].to_numpy(), label=str(i), histtype="stepfilled", alpha=1, colour=color) for i, color in zip(range(2,11), sns.color_palette("hls", 9))]

    # Initialise histogram plot
    plot_histo = HistogramPlot(
        ylabel="Number of jets",
        xlabel="$p_{\mathrm{T}}$ [TeV]",
        logy=True,
        # bins=np.linspace(0, 5, 60),  # you can force a binning for the plot here
        bins=200,  # you can also define an integer number for the number of bins
        bins_range=(0.04, 4.6),  # only considered if bins is an integer
        ymin=1e-5,
        ymax=1e12,
        norm=False,
        atlas_first_tag="Simulation Internal",
        figsize=(6, 5),
        n_ratio_panels=0,
        draw_errors=False,
    )

    # Add histograms and plot
    for hist in hists:
        plot_histo.add(hist)
    plot_histo.draw()

    plot_histo.savefig("physical.pdf")
    plot_histo.savefig("physical.png")
    plot_histo.savefig("physical.jpg")
    plt.cla()
    plt.clf()
    plt.close()
    
def sns_plotting(df):
    pt_name = r"$p_{\mathrm{T}}$ [TeV]"
    slice_name = "JZ"
    df = df.rename(columns={"jets_pt": pt_name, "JZ_slice": slice_name})
    ax = sns.histplot(data=df, x=pt_name, hue=slice_name, weights="weight", multiple="stack", log_scale=(False, True), bins=100, common_norm=True, stat="count", palette=sns.color_palette("hls", 9), element="step", edgecolor=None)
    sns.move_legend(ax, "upper right", ncol=2, title="JZ slice")
    # h,l = plot.axes.get_legend_handles_labels()
    # plot.axes.legend_.remove()
    # plot.legend(h,l, ncol=2)
    # sat the legend to have 2 columns
    plt.xlabel(r"$p_{\mathrm{T}}$ [TeV]")
    plt.ylabel(r"Number of jets")
    plt.ylim(1e-6, 6e12)
    plt.xlim(0.04, 5.)
    atlasify.atlasify(
        axes=ax,
        subtext='Simulation Internal',
    )
    plt.savefig("physical_histplot.pdf")
    plt.savefig("physical_histplot.png", dpi=400)
    plt.cla()
    plt.clf()
    plt.close()
    
def sns_label_plotting(df):
    pt_name = r"$p_{\mathrm{T}}$ [TeV]"
    label_name = "label"
    df = df.rename(columns={"jets_pt": pt_name, "jets_PartonTruthLabelID": label_name})
    ax = sns.histplot(data=df, x=pt_name, hue=label_name, weights="weight", multiple="layer", log_scale=(False, True), bins=100, common_norm=False, stat="count", palette='Set1', element="step")
    # sns.move_legend(ax, "upper right", ncol=2, title="JZ slice")
    # h,l = plot.axes.get_legend_handles_labels()
    # plot.axes.legend_.remove()
    # plot.legend(h,l, ncol=2)
    # sat the legend to have 2 columns
    plt.xlabel(r"$p_{\mathrm{T}}$ [TeV]")
    plt.ylabel(r"Number of jets")
    plt.ylim(1e-6, 6e12)
    plt.xlim(0.04, 5.)
    atlasify.atlasify(
        axes=ax,
        subtext='Simulation Internal',
    )
    plt.savefig("physical_histplot_label.pdf")
    plt.savefig("physical_histplot_label.png", dpi=400)
    plt.cla()
    plt.clf()
    plt.close()
    
if __name__ == "__main__":
    HUE_MAPPER = {1: 'quark', 2: 'quark', 3: 'quark', 4: 'quark', 5: 'quark', 6: 'quark', 21: 'gluon'}
    df = pd.read_csv("data/pythia_physical.csv")
    df['jets_pt'] *= 1e-6
    # remap the labels
    df['jets_PartonTruthLabelID'] = df['jets_PartonTruthLabelID'].replace(HUE_MAPPER)
    sns_label_plotting(df)
    sns_plotting(df)