import os
import sys
sys.path.append(os.getcwd())
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import atlasify
import argparse
import puma

from jidenn.const import MC_NAMING_SCHEMA

parser = argparse.ArgumentParser()
parser.add_argument("--load_path", type=str, help="Path to the root file")

var_mapping = {
    "jets_pt": r"$p_{\mathrm{T}}$ [GeV]",
    "jets_eta": r"$\eta$",
    "jets_TopoTower_n+jets_Constituent_n": r"$N_{\mathrm{PFO}} + N_{\mathrm{TopoTower}}$",
}

def sns_label_plotting_phys(df, save_path, var, subtext=''):
    # N_name = r"$p_{\mathrm{T}}$ [TeV]"
    # N_name = r"$N_{\mathrm{PFO}} + N_{\mathrm{TopoTower}}$"
    N_name = var_mapping[var]
    label_name = "label"
    df = df.rename(columns={var: N_name, "jets_PartonTruthLabelID": label_name})
    ax = sns.histplot(data=df, x=N_name, hue=label_name, weights="weight", multiple="layer", log_scale=(
        False, False), bins=np.linspace(0,100,100), common_norm=False, stat="count", palette='Set1', element="step", fill=False, hue_order=['quark', 'gluon'])
    ax.legend_.set_title(None)

    lss = ['-', '--']
    # handles = ax.legend_.legendHandles[::-1]
    # for line, ls, handle in zip(ax.lines, lss, handles):
    #     line.set_linestyle(ls)
    #     handle.set_ls(ls)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    
    plt.setp(ax.get_legend().get_texts(), fontsize='15')
    plt.xlabel(N_name, horizontalalignment='right', x=1.0, fontsize=15)
    plt.ylabel(r"Counts", horizontalalignment='right', y=1.0, fontsize=15)
    # add vertical line at 60
    
    plt.axvline(x=60, color='green', linestyle='-', linewidth=2)
    # plt.ylim(1e-1, 6e9)
    plt.xlim(0, 100)
    atlasify.atlasify(
        axes=ax,
        subtext='13 TeV, Pythia8\n' + r'anti-$k_{\mathrm{T}}$, $R = 0.4$ PFlow jets' + '\n' + subtext,
        atlas='Simulation Internal',
        font_size=15,
        sub_font_size=15,
        label_font_size=15,
    )
    plt.savefig(f"{save_path}/n_topo+n_const.pdf", bbox_inches='tight')
    plt.savefig(f"{save_path}/n_topo+n_const.png", dpi=400, bbox_inches='tight')
    plt.close()
    
COLOR_PALETTE = [sns.color_palette('colorblind'), sns.color_palette("Set1")]
MARKER_PALETTE = ['o', 's', 'v', 'D', 'P', 'X', 'd',
                   'p', 'h', '8', '>', '<', '^', '*', '+', '8']
LINESTYLE_PALETTE = ['-', '--', '-.', ':']

def sns_label_plotting_phys2(
            df: pd.DataFrame, 
            err_df: pd.DataFrame,
            bins: np.ndarray,
            save_path: str | list[str],
            ylabel: str = "Normalised counts",
            xlabel: str | None = None,
            ylog: bool = False,
            xlog: bool = False,
            norm: bool = True,
            second_tag: str = "",
            include_errors: bool = True,
            atlas_first_tag: str = "",
            color_pallete: int | None = 1,
            ratio_reference: str | None = None,
            ymax_ratio: list[float] | None = None,
            ymin_ratio: list[float] | None = None,
            **kwargs):
    
    color = COLOR_PALETTE[color_pallete] 
    plot_histo = puma.HistogramPlot(
            n_ratio_panels=1,
            ylabel=ylabel,
            xlabel=xlabel,
            logy=ylog,
            logx=xlog,
            ymin_ratio=ymin_ratio,
            ymax_ratio=ymax_ratio,
            leg_ncol=2,
            atlas_first_tag=atlas_first_tag,
            atlas_second_tag=second_tag,
            draw_errors=include_errors,
            norm=norm,
            bins=bins,
            **kwargs
        )
    for i, col in enumerate(df.columns):
        values = df[col].values
        err_values = err_df[col].values if include_errors else None
            
        plot_histo.add(puma.Histogram(values=values,
                                    sum_squared_weights=err_values**2,
                                    bin_edges=bins,
                                    colour=color[i % len(color)],
                                    # linestyle=LINESTYLE_PALETTE[i % len(LINESTYLE_PALETTE)],
                                    is_data=False,
                                    markersize=None,
                                    label=col,),
                    key=col,
                    reference=True if ratio_reference == col else False)

    
    plot_histo.draw()
    for path in save_path if isinstance(save_path, list) else [save_path]:
        plot_histo.savefig(path, dpi=300)
    plt.close()


def load_dataframe(path, vars):
    from jidenn.data.JIDENNDataset import JIDENNDataset
    ds = JIDENNDataset.load(path)
    ds = ds.filter(lambda x: x['jets_TopoTower_n'] > 1)
    ds = ds.remap_data(lambda x: {var: x[var] for var in vars}).to_pandas()
    # ds = ds.to_pandas()
    return ds


def main():
    HUE_MAPPER = {1: 'quark', 2: 'quark', 3: 'quark', 4: 'quark', 5: 'quark', 6: 'quark', 21: 'gluon'}
    # args = parser.parse_args()
    # phys_path = args.load_path
    phys_path = "data/r22_PFO/fwd_phys_20-2500GeV/Pythia8EvtGen_A14NNPDF23LO_jetjet"
    
    try:    
        df = pd.read_csv(os.path.join(phys_path, "pythia_physical2.csv"))
    except FileNotFoundError:
        df = load_dataframe(phys_path, ["jets_pt", "jets_eta", "jets_PartonTruthLabelID", "weight", "jets_Constituent_n", "jets_TopoTower_n"])
        df['jets_TopoTower_n+jets_Constituent_n'] = df['jets_TopoTower_n'] + df['jets_Constituent_n']
        df['jets_PartonTruthLabelID'] = df['jets_PartonTruthLabelID'].replace(HUE_MAPPER)
        df.to_csv(os.path.join(phys_path, "pythia_physical2.csv"))
    # df['jets_pt'] *= 1e-6
    frac = df.query('jets_TopoTower_n+jets_Constituent_n > 60 & jets_TopoTower_n > 0')['weight'].sum()/df.query('jets_TopoTower_n > 0')['weight'].sum()
    print(f"Fraction of events with N_PFO + N_TopoTower > 60: {frac*100:.1f}%")
    sns_label_plotting_phys(df, phys_path, var="jets_TopoTower_n+jets_Constituent_n", subtext=r"Frac. above 60: " +f"{frac*100:.0f}%")
    # print('Done!')
    
def main2():
    proc = ['Pythia8EvtGen_A14NNPDF23LO_jetjet', 'PhPy8EG_A14_ttbar_hdamp258p75_nonallhad', 
            'PowhegPy8EG_NNLOPS_nnlo_30_ggH125_tautaul13l7', 'PowhegPy8EG_NNPDF30_AZNLOCTEQ6L1_VBFH125_tautaul13l7', 
            'PowhegPythia8EvtGen_AZNLOCTEQ6L1_Zmumu', 'Sherpa_222_NNPDF30NNLO_SinglePhoton']
    dfs = []
    dfs_err = []
    bins = np.linspace(20, 200, 10)
    for p in proc:
        path = f'data/r22_PFO/central_2d_phys_200GeV/{p}/distrib.csv'
        df = pd.read_csv(path)
        df['jets_pt'] *= 1e-3
        df['jets_pt_bin'] = pd.cut(df['jets_pt'], bins=bins)
        df['error'] = df['weight']**2
        df['label'] = df['jets_PartonTruthLabelID'].replace({1: 'quark', 2: 'quark', 3:'quark', 4:'quark', 5:'quark', 6:'quark', 21:'gluon'})
        
        df_q = df.query('label == "quark"').groupby('jets_pt_bin').agg({'weight': 'sum', 'error': 'sum'}).reset_index()
        df_g = df.query('label == "gluon"').groupby('jets_pt_bin').agg({'weight': 'sum', 'error': 'sum'}).reset_index()
        df_q['error'] = np.sqrt(df_q['error'])
        df_g['error'] = np.sqrt(df_g['error'])
        df = pd.merge(df_q, df_g, on='jets_pt_bin', suffixes=('_q', '_g'))
        df = df.set_index('jets_pt_bin')
        df.index.name = 'jets_pt'
        df['frac'] = df['weight_q']/df['weight_g']
        df['frac_err'] = df['frac']*np.sqrt((df['error_q']/df['weight_q'])**2 + (df['error_g']/df['weight_g'])**2)
        # df = df[['frac', 'frac_err']].rename(columns={'frac': MC_NAMING_SCHEMA[p], 'frac_err': MC_NAMING_SCHEMA[p]+'_err'})
        dfs.append(df[['frac']].rename(columns={'frac': MC_NAMING_SCHEMA[p]}))
        dfs_err.append(df[['frac_err']].rename(columns={'frac_err': MC_NAMING_SCHEMA[p]}))
    # merge the dataframes
    df = pd.concat(dfs, axis=1)
    dfs_err = pd.concat(dfs_err, axis=1)
    print(df)
    print(dfs_err)
    
    sns_label_plotting_phys2(df, dfs_err, bins=bins, save_path=['data/r22_PFO/central_2d_phys_200GeV/frac.pdf'],
                             ylabel='Quark/Gluon ratio', xlabel=r'$p_{\mathrm{T}}$ [GeV]', ylog=True, xlog=False, norm=False, include_errors=True, atlas_first_tag='Simulation Internal', color_pallete=1, ratio_reference=MC_NAMING_SCHEMA['Pythia8EvtGen_A14NNPDF23LO_jetjet'])
    
    

if __name__ == "__main__":
    main()