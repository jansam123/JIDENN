import os
import sys
sys.path.append(os.getcwd())
from typing import Optional, Callable
from functools import partial
import pandas as pd
import atlasify
import matplotlib.pyplot as plt
import seaborn as sns

from jidenn.const import MC_NAMING_SCHEMA, METRIC_NAMING_SCHEMA, MODEL_NAMING_SCHEMA


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
    HIGHLIGHT_DIAGONAL = False
    HIGHLIGHT_BEST = True
    # XLABEL = 'Training MC'
    XLABEL = 'Model'
    YLABEL = 'Evaluation MC'
    LOGDIR = 'augmentations'
    SAVEPATH = f'/home/jankovys/JIDENN/logs/augmentations/evaluation/50wp/2D_aug_model_{METRIC}'

    # subtext=r'13 TeV, anti-$k_{\mathrm{T}}$, $R = 0.4$ PFlow jets' + '\n' + r'DeParT, 50% WP, $p_{\mathrm{T}} > 400$ GeV'
    # SQUARE=False
    # figsize=(9, 5)
    # model_names = ['depart', 'depart-Sherpa_CT10_CT14nnlo_CSShower_Lund_2to2jets', 'depart-Sh_2211_Enh_clusterTune',
    #                'depart-Sh_2211_jj_DIRE', 'depart-H7EG_jetjet_Cluster', 'depart-H7EG_jetjet_Cluster_dipole', 'depart-multiMC']

    # model_labels = ['Py', 'Sh St.', 'Sh Cl.',
    #                 'Sh Dire', 'Hw Ang.', 'Hw Dip.', 'Multi']

    subtext = r'13 TeV, anti-$k_{\mathrm{T}}$, $R = 0.4$ PFlow jets' + \
        '\n' + r'50% WP, $p_{\mathrm{T}} > 400$ GeV'
    SQUARE = True
    figsize = (12, 6)
    model_names = ['depart', 'depart-all_aug', 'depart-coll_split', 'depart-pt_smear', 'depart-rot_drop_smear', 'depart-rotation',
                   'depart-shift_weights', 'depart-soft_drop', 'depart-soft_smear', 'pfn', 'fc_crafted', 'depart-ircs', 'efn', 'depart-Sherpa_CT10_CT14nnlo_CSShower_Lund_2to2jets', 'depart-Sh_2211_Enh_clusterTune',
                   'depart-Sh_2211_jj_DIRE', 'depart-H7EG_jetjet_Cluster', 'depart-H7EG_jetjet_Cluster_dipole', 'depart-multiMC', 'pfn-multiMC', 'fc_crafted-multiMC']
    model_labels = ['DeParT', 'DeParT [all]', 'DeParT [coll. split]', 'DeParT [pT smear]', 'DeParT [rot+drop+smear]', 'DeParT [rot.]', 'DeParT [shift w.]',
                    'DeParT [soft drop]', 'DeParT [soft smear]', 'PFN', 'FC red.', 'DeParT IRCS', 'EFN', 'DeParT Sh St.', 'DeParT Sh Cl.',
                    'DeParT Sh Dire', 'DeParT Hw Ang.', 'DeParT Hw Dip.', 'DeParT MultiMC', 'PFN MultiMC', 'FC red. MultiMC']

    dir_names = ['Pythia8EvtGen_A14NNPDF23LO_jetjet', 'Sherpa_CT10_CT14nnlo_CSShower_Lund_2to2jets', 'Sh_2211_Enh_clusterTune',
                 'Sh_2211_jj_DIRE', 'H7EG_jetjet_Cluster', 'H7EG_jetjet_Cluster_dipole']
    dir_labels = ['Py', 'Sh St.', 'Sh Cl.',
                  'Sh Dire', 'Hw Ang.', 'Hw Dip.']
    
    NOMINAL_LABEL = 'Py'
    if INCLUDE_NLO:
        dir_names += ['PhH7EG_jj', 'PhPy8EG_jj']
        dir_labels += ['PhHw', 'PhPy']

    model_mapping = dict(zip(model_names, model_labels))
    dir_mapping = dict(zip(dir_names, dir_labels))

    def dir_and_model_name_to_path_map(dir_name, model_name):
        return f'/home/jankovys/JIDENN/logs/{LOGDIR}/evaluation/50wp/{dir_name}/models/{model_name}/binned_metrics.csv'

    df = load_multiple_dirs(dir_names, model_names,
                            dir_and_model_name_to_path_map)

    df['model'] = df['model'].replace(model_mapping)
    df['dir'] = df['dir'].replace(dir_mapping)

    df = df.query('bin_mid > 0.4')
    df = df[['model', 'dir', METRIC]]
    df = df.groupby(['model', 'dir']).mean().reset_index()

    # divide by nominal value
    scatter_df = df.copy()
    scatter_df = scatter_df.merge(df.query(f'dir == "{NOMINAL_LABEL}"')[['model', METRIC]],
                                  on='model', suffixes=('', '_nominal'))
    scatter_df[METRIC] = scatter_df[METRIC]/scatter_df[METRIC+'_nominal']
    #
    # take max value
    scatter_df = scatter_df.groupby(['model']).min().reset_index()
    scatter_df[METRIC] = 1-scatter_df[METRIC]
    scatter_df = scatter_df[['model', METRIC, METRIC+'_nominal']]
    # order by metric_nominal
    scatter_df = scatter_df.sort_values(METRIC+'_nominal', ascending=False)
    # make a 2D plot, where on the x axis is the models nominal value and on the y axis is the models value
    # put the legend on the right side
    # give each model a different marker
    # make the markers bigger
    sns.scatterplot(data=scatter_df, x=METRIC+'_nominal',
                    y=METRIC, hue='model', s=200, legend='full', style='model', edgecolor='black', linewidth=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('Nominal performance')
    plt.ylabel('MC dependence')
    atlasify.atlasify(atlas="Simulation Internal",
                      outside=False,
                      subtext=subtext,
                      )
    plt.savefig(SAVEPATH+'_scatter.png', dpi=400, bbox_inches='tight')
    plt.savefig(SAVEPATH+'_scatter.pdf', bbox_inches='tight')
    # increase marker size
    plt.clf()

    sns.scatterplot(data=scatter_df, x=METRIC+'_nominal',
                    y=METRIC, hue='model', s=280, legend='full', style='model', edgecolor='black', linewidth=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('Nominal performance')
    plt.ylabel('MC dependence')
    atlasify.atlasify(atlas="Simulation Internal",
                      outside=False,
                      subtext=subtext,
                      )
    plt.xlim(19, 22.25)
    plt.ylim(0.44, 0.54)
    plt.savefig(SAVEPATH+'_scatter_zoom.png', dpi=400, bbox_inches='tight')
    plt.savefig(SAVEPATH+'_scatter_zoom.pdf', bbox_inches='tight')

    # df[METRIC] = df[METRIC]/df[METRIC].max()
    df = df.pivot(index='dir', columns='model', values=METRIC)
    # order Hw Ang.    Hw Dip.         Py     Sh Cl.    Sh Dire     Sh St. coulms to Py, Sh St., Sh Cl., Sh Dire, Hw Ang.    Hw Dip.
    df = df[model_labels]
    df = df.loc[dir_labels]

    fig = plt.figure(figsize=figsize)
    ax = sns.heatmap(df, cmap="flare", annot=True, fmt='.1f',
                     square=SQUARE)
    # for i, spine in ax.spines.items():
    #     spine.set_visible(True)
    cbar = ax.collections[0].colorbar
    cbar.set_label(METRIC_NAMING_SCHEMA[METRIC])

    if HIGHLIGHT_DIAGONAL:
        for i in range(len(model_names)-1):
            ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                                       edgecolor='black', lw=1, clip_on=False))
    if HIGHLIGHT_BEST:
        for i in range(len(df)):
            best = df.iloc[i].idxmax()
            best_idx = df.columns.get_loc(best)
            ax.add_patch(plt.Rectangle((best_idx, i), 1, 1, fill=False,
                                       edgecolor='green', lw=1, clip_on=False))

    plt.xlabel(XLABEL, horizontalalignment='right',
               x=1.0, fontsize=13)
    plt.ylabel(YLABEL, horizontalalignment='right',
               y=1.0, fontsize=13)
    atlasify.atlasify(atlas="Simulation Internal",
                      outside=True,
                      subtext=subtext,
                      minor_tick_length=0,
                      major_tick_length=2,
                      # font_size=fontsize,
                      # sub_font_size=fontsize,
                      # label_font_size=fontsize,
                      )
    plt.savefig(SAVEPATH+'.png', dpi=400, bbox_inches='tight')
    plt.savefig(SAVEPATH+'.pdf', bbox_inches='tight')

    print(df)


if __name__ == '__main__':
    main()
