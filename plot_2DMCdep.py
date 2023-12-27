from typing import Optional, Callable
from functools import partial
import numpy as np
import pandas as pd
import atlasify
import matplotlib.pyplot as plt
import seaborn as sns

from jidenn.const import MC_NAMING_SCHEMA, METRIC_NAMING_SCHEMA


def plot_2d(array_2d: np.ndarray,
            mask: np.ndarray,
            clabel: str,
            save_path: str,
            subtext: str = '',
            max_const: int = 50,
            vmin: Optional[float] = None,
            vmax: Optional[float] = None,
            fontsize: int = 20,) -> None:

    ax = sns.heatmap(array_2d, mask=mask, vmin=vmin, vmax=vmax)
    ticks = [i for i in range(0, max_const, 10)] + [max_const]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_yticklabels(ticks)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.tick_params(axis='both', which='minor', labelsize=fontsize)
    ax.invert_yaxis()
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.set_label(clabel, fontsize=fontsize)
    plt.xlabel('Constituent Index', horizontalalignment='right',
               x=1.0, fontsize=fontsize)
    plt.ylabel('Constituent Index', horizontalalignment='right',
               y=1.0, fontsize=fontsize)

    atlasify.atlasify(atlas="Simulation Internal",
                      outside=True,
                      subtext=subtext,
                      font_size=fontsize,
                      sub_font_size=fontsize,
                      label_font_size=fontsize,
                      )
    plt.savefig(f'{save_path}.png', dpi=400, bbox_inches='tight')
    plt.savefig(f'{save_path}.pdf', bbox_inches='tight')
    plt.close()


def load_saved_metrics(names: list, name_to_path_map: Callable[[str], str]) -> pd.DataFrame:
    dfs = [pd.read_csv(name_to_path_map(name), index_col=0) for name in names]
    for df, name in zip(dfs, names):
        df['model'] = name
    return pd.concat(dfs, ignore_index=True)


def load_multiple_dirs(dir_names: list, model_names: list, dir_and_model_name_to_path_map: Callable[[str, str], str]) -> pd.DataFrame:
    dfs = []
    for dir_name in dir_names:
        dir_df = load_saved_metrics(model_names, partial(
            dir_and_model_name_to_path_map, dir_name))
        dir_df['dir'] = dir_name
        dfs.append(dir_df)
    return pd.concat(dfs, ignore_index=True)


def main():
    METRIC = 'gluon_rejection'
    INCLUDE_NLO = False
    model_names = ['depart', 'depart-Sh_2211_Enh_clusterTune', 'depart-H7EG_jetjet_Cluster', 'depart-H7EG_jetjet_Cluster_dipole',
                   'depart-Sherpa_CT10_CT14nnlo_CSShower_Lund_2to2jets', 'depart-Sh_2211_jj_DIRE', 'depart-multiMC']

    dir_names = ['Sh_2211_Enh_clusterTune', 'H7EG_jetjet_Cluster', 'H7EG_jetjet_Cluster_dipole',
                 'Sherpa_CT10_CT14nnlo_CSShower_Lund_2to2jets', 'Sh_2211_jj_DIRE', 'Pythia8EvtGen_A14NNPDF23LO_jetjet']
    dir_names += ['PhH7EG_jj', 'PhPy8EG_jj'] if INCLUDE_NLO else []

    def dir_and_model_name_to_path_map(
        dir_name, model_name): return f'/home/jankovys/JIDENN/logs/augmentations/evaluation/50wp/{dir_name}/models/{model_name}/binned_metrics.csv'
    df = load_multiple_dirs(dir_names, model_names,
                            dir_and_model_name_to_path_map)
    df['model'] = df['model'].str.replace(
        'depart-', '').replace('depart', 'Pythia8EvtGen_A14NNPDF23LO_jetjet')
    df['model'] = df['model'].replace(MC_NAMING_SCHEMA)
    df['dir'] = df['dir'].replace(MC_NAMING_SCHEMA)
    df = df.query('bin_mid > 0.4')
    df = df[['model', 'dir', METRIC]]
    df = df.groupby(['model', 'dir']).mean().reset_index()
    # df[METRIC] = df[METRIC]/df[METRIC].max()
    df = df.pivot(index='dir', columns='model', values=METRIC)
    # order Hw Ang.    Hw Dip.         Py     Sh Cl.    Sh Dire     Sh St. coulms to Py, Sh St., Sh Cl., Sh Dire, Hw Ang.    Hw Dip.
    ordered = ['Py', 'Sh St.', 'Sh Cl.',
               'Sh Dire', 'Hw Ang.', 'Hw Dip.']
    df = df.loc[ordered + ['PhPy', 'PhHw'] if INCLUDE_NLO else ordered]
    # order the same way indicies
    df = df[ordered+['Multi']]
    ax = sns.heatmap(df, cmap="flare", annot=True, fmt='.1f',
                     square=True if not INCLUDE_NLO else False)
    # for i, spine in ax.spines.items():
    #     spine.set_visible(True)
    cbar = ax.collections[0].colorbar
    cbar.set_label(METRIC_NAMING_SCHEMA[METRIC])

    for i in range(len(model_names)-1):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                                   edgecolor='black', lw=1, clip_on=False))
    for i in range(len(df)):
        best = df.iloc[i].idxmax()
        best_idx = df.columns.get_loc(best)
        ax.add_patch(plt.Rectangle((best_idx, i), 1, 1, fill=False,
                                   edgecolor='green', lw=1, clip_on=False))

    plt.xlabel('Training MC', horizontalalignment='right',
               x=1.0, fontsize=13)
    plt.ylabel('Evaluation MC', horizontalalignment='right',
               y=1.0, fontsize=13)
    atlasify.atlasify(atlas="Simulation Internal",
                      outside=True,
                      subtext=r'13 TeV, anti-$k_{\mathrm{T}}$, $R = 0.4$ PFlow jets' + '\n' +
                      r'DeParT, 50% WP, $p_{\mathrm{T}} > 400$ GeV',
                      minor_tick_length=0,
                      major_tick_length=2,
                      # font_size=fontsize,
                      # sub_font_size=fontsize,
                      # label_font_size=fontsize,
                      )
    plt.savefig(
        f'/home/jankovys/JIDENN/logs/augmentations/evaluation/50wp/2D_MC_model.png', dpi=400, bbox_inches='tight')
    plt.savefig(
        f'/home/jankovys/JIDENN/logs/augmentations/evaluation/50wp/2D_MC_model.pdf', bbox_inches='tight')

    print(df)


if __name__ == '__main__':
    main()
