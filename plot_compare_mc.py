import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import atlasify
import argparse
import puma
#
from jidenn.evaluation.plotter import plot_var_dependence
from jidenn.const import METRIC_NAMING_SCHEMA, LATEX_NAMING_CONVENTION, MODEL_NAMING_SCHEMA, MC_NAMING_SCHEMA

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--load_dir", default=".", type=str,
                    help="Directory to load the metrics from.")
parser.add_argument("--summary_csv", default=".", type=str,
                    help="Path to the summary csv.")
parser.add_argument("--model_names", nargs='*', type=str,
                    help="names of the models.")
parser.add_argument("--save_dir", default=".", type=str,
                    help="Directory to save the plots to.")
parser.add_argument("--type", default="pT", type=str, help="Type of the plot.")
parser.add_argument("--compare_mc", default=False, type=bool,
                    help="Compare MC models.")


def main(args: argparse.Namespace):

    [plot_var_dependence(dfs=[pd.read_csv(os.path.join(dir, 'models', f'{model}', 'binned_metrics.csv')) for dir in args.load_dirs],
                         labels=[MC_NAMING_SCHEMA[label] for label in args.labels],
                         bin_midpoint_name='bin_mid',
                         bin_width_name='bin_width',
                         metric_names=args.metric_names,
                         save_path=os.path.join(args.save_dir, 'models', f'{model}'),
                         ratio_reference_label=MC_NAMING_SCHEMA[args.reference],
                         xlabel=r'$p_T$ [TeV]',
                         ylabel_mapper=METRIC_NAMING_SCHEMA,
                         ylims=args.ylims1,
                         figsize=(8, 6),
                         xlog=args.xlog,
                         h_line_position=[None, args.wp_val, None],
                         leg_loc='upper right',
                         title=f'{MODEL_NAMING_SCHEMA[model]}, {args.wp} WP',
                         colours=sns.color_palette("Set1", len(args.labels))) for model in args.model_names]

    for metric in ['quark_efficiency', 'gluon_rejection']:
        df_big = pd.DataFrame()
        overall_metrics = pd.read_csv(os.path.join(args.ref_dir, 'overall_metrics.csv'), index_col=0)
        acc_sorted_models = overall_metrics.sort_values(by='binary_accuracy', ascending=False).index

    overall_metrics = pd.read_csv(os.path.join(args.ref_dir, 'overall_metrics.csv'), index_col=0)
    acc_sorted_models = overall_metrics.sort_values(by='binary_accuracy', ascending=False).index

    for label, dir in zip(args.labels, args.load_dirs):
        dfs = []
        for model in args.model_names:
            ref_df = pd.read_csv(os.path.join(args.ref_dir, 'models', f'{model}', 'binned_metrics.csv'))
            if model == args.reference:
                continue
            df = pd.read_csv(os.path.join(dir, 'models', f'{model}', 'binned_metrics.csv'))
            df['ratio'] = df[metric] / ref_df[metric]
            df['diff'] = ref_df[metric] - df[metric]
            dfs.append(df)
            df['mc'] = label
            df['model'] = model
            df_big = pd.concat([df_big, df])

            labels, dfs = zip(*sorted(zip(args.model_names, dfs), key=lambda x: acc_sorted_models.get_loc(x[0])))
            labels = [MODEL_NAMING_SCHEMA[model] for model in labels]

            plot_var_dependence(dfs=dfs,
                                labels=labels,
                                bin_midpoint_name='bin_mid',
                                bin_width_name='bin_width',
                                metric_names=['diff', 'ratio'],
                                save_path=os.path.join(args.save_dir, 'MCs', f'{label}', metric),
                                ratio_reference_label=None,
                                xlabel=r'$p_T$ [TeV]',
                                ylabel_mapper={'diff': f'Difference fromn Pythia', 'ratio': f'Ratio'},
                                ylims=args.ylims2,
                                # ylims = None,
                                xlog=args.xlog,
                                figsize=(10, 4),
                                h_line_position=[0.0, 1.0],
                                leg_loc='upper center',
                                title=f'{args.wp}, {MC_NAMING_SCHEMA[label]}',
                                colours=sns.color_palette("coolwarm", len(args.model_names)))

        # df_had_difff =

        def calc_diff(df_big):
            df = pd.DataFrame()
            df['had_diff'] = (df_big[df_big['mc'] == 'sherpa'][metric] - df_big[df_big['mc'] ==
                                                                                'sherpa_lund'][metric]) / df_big[df_big['mc'] == 'pythia'][metric]
            df['ps_diff'] = (df_big[df_big['mc'] == 'herwig7'][metric] - df_big[df_big['mc'] ==
                             'herwig7_dipole'][metric]) / df_big[df_big['mc'] == 'pythia'][metric]
            df['ps_diff_sherpa'] = (df_big[df_big['mc'] == 'sherpa'][metric] - df_big[df_big['mc'] ==
                                                                                      'sherpa_dire'][metric]) / df_big[df_big['mc'] == 'pythia'][metric]
            return df

        df_diff = df_big.copy()[['model', 'bin_mid', 'bin_width', 'mc', metric]].groupby(
            ['model', 'bin_mid', 'bin_width']).apply(calc_diff).reset_index()
        df_en = df_big.copy()[['model', 'bin_mid', 'bin_width', 'ratio', 'diff']]
        df_en['ratio'] = (df_en['ratio'] - 1).abs()
        df_en['diff'] = df_en['diff'].abs()

        df_envel = df_en.groupby(['model', 'bin_mid', 'bin_width']).mean().reset_index()
        df_envel = df_envel.rename(columns={'ratio': 'ratio_mean', 'diff': 'diff_mean'})

        df_envel2 = df_en.groupby(['model', 'bin_mid', 'bin_width']).max().reset_index()
        df_envel2 = df_envel2.rename(columns={'ratio': 'ratio_max', 'diff': 'diff_max'})

        df_envel = df_envel.merge(df_envel2, on=['model', 'bin_mid', 'bin_width'])
        df_envel = df_envel.merge(df_diff, on=['model', 'bin_mid', 'bin_width'])
        dfs_eval = []
        labels = []
        for model in df_envel['model'].unique():
            df = df_envel[df_envel['model'] == model]
            labels.append(model)
            dfs_eval.append(df)

        labels, dfs_eval = zip(*sorted(zip(labels, dfs_eval), key=lambda x: acc_sorted_models.get_loc(x[0])))
        labels = [MODEL_NAMING_SCHEMA[model] for model in labels]
        model_label = r'\varepsilon^{-1}_{g,\mathrm{model}}' if metric == 'gluon_rejection' else r'\varepsilon_{q,\mathrm{model}}'
        py_label = r'\varepsilon^{-1}_{g,\mathrm{Pythia}}' if metric == 'gluon_rejection' else r'\varepsilon_{q,\mathrm{Pythia}}'
        diff_label = r'$\varepsilon^{-1}_{g}$' if metric == 'gluon_rejection' else r'$\varepsilon_{q}$'
        plot_var_dependence(dfs=dfs_eval,
                            labels=labels,
                            bin_midpoint_name='bin_mid',
                            bin_width_name='bin_width',
                            metric_names=['ratio_mean', 'diff_mean', 'ratio_max',
                                          'diff_max', 'had_diff', 'ps_diff', 'ps_diff_sherpa'],
                            save_path=os.path.join(args.save_dir, 'envelope', metric),
                            ratio_reference_label=None,
                            xlabel=r'$p_T$ [TeV]',
                            ylabel_mapper={'ratio_mean': r'Envelope of models, mean $\left(\frac{' + model_label + r'}{' + py_label + r'}\right)$',
                                           'diff_mean': r'Envelope of models, mean $\left(' + model_label + r' - ' + py_label + r'\right)$',
                                           'ratio_max': r'Envelope of models, max $\left(\frac{' + model_label + r'}{' + py_label + r'}\right)$',
                                           'diff_max': r'Envelope of models, max $\left(' + model_label + r' - ' + py_label + r'\right)$',
                                           'had_diff': f'{diff_label} difference, (Cluster - String Had.) / Pythia',
                                           'ps_diff': f'{diff_label} difference, (Ang. ord. - Dipole PS) / Pythia',
                                           'ps_diff_sherpa': f'{diff_label} difference, (CSS (dipole) - DIRE PS) / Pythia'
                                           },
                            h_line_position=[None, None, None, None, 0.0, 0.0, 0.0],
                            xlog=args.xlog,
                            figsize=(7, 5),
                            ylims=[None, None, (0.03, 0.2), None, (-0.16, 0.16), (-0.16, 0.16), None] if metric == 'quark_efficiency' else [
                                None, None, (0.1, 0.8), None, (-0.25, 0.25), (-0.2, 0.7), None],
                            leg_loc='upper right',
                            title=[f'{args.wp} WP', f'{args.wp} WP', f'{args.wp} WP', f'{args.wp} WP',
                                   f'{args.wp} WP, Sherpa2.2.5', f'{args.wp} WP, Herwig7', f'{args.wp} WP, Sherpa2.2.11'],
                            colours=sns.color_palette("coolwarm", len(args.model_names)))

        df_big = df_big[['mc', 'model', metric]]
        df_big = df_big.groupby(['mc', 'model']).mean().reset_index()
        df_big['model'] = df_big['model'].map(MODEL_NAMING_SCHEMA)
        df_big['mc'] = df_big['mc'].map(MC_NAMING_SCHEMA)
        df_big = df_big.rename(columns={'mc': 'MC Sample', 'model': 'Deep Learning Model'})
        fig = plt.figure(figsize=(14, 6))
        ax = sns.histplot(data=df_big, hue='MC Sample', x='Deep Learning Model',
                          multiple='dodge', palette='Set1', weights=metric,
                          shrink=.8, hue_order=[MC_NAMING_SCHEMA[mc] for mc in args.mc_names])

        ax.legend_.set_title(None)
        plt.ylim(0., 0.8)
        plt.ylabel(METRIC_NAMING_SCHEMA[metric])
        atlasify.atlasify(
            atlas=True,
            subtext="Simulation Internal \n 13 TeV",
        )
        plt.savefig(os.path.join(args.save_dir, f'summary_hist.png'), dpi=400, bbox_inches='tight')


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # mc comparison
    # wp = '50'
    for wp_val in [0.5]:
        print(wp_val)
        args.wp_val = wp_val
        wp = str(int(wp_val * 100))
        args.wp = wp
        load_dir = f'/home/jankovys/JIDENN/logs/pythia_Wflat_JZ10/evaluation/{wp}wp'
        args.save_dir = f'/home/jankovys/JIDENN/logs/pythia_Wflat_JZ10/evaluation/{wp}wp/comparison'
        args.mc_names = ['pythia', 'herwig7_dipole', 'sherpa_dire', 'herwig7',
                         'sherpa_enh_cluster_tune', 'sherpa', 'sherpa_lund', 'powheg+pythia',
                         'powheg+herwig7']
        args.load_dirs = ['Pythia8EvtGen_A14NNPDF23LO_jetjet', 'H7EG_jetjet_Cluster_dipole', 'Sh_2211_jj_DIRE', 'H7EG_jetjet_Cluster',
                          'Sh_2211_Enh_clusterTune', 'Sherpa_CT10_CT14nnlo_CSShower_2to2jets', 'Sherpa_CT10_CT14nnlo_CSShower_Lund_2to2jets',
                          'PhPy8EG_jj', 'PhH7EG_jj']
        args.load_dirs = [os.path.join(load_dir, mc) for mc in args.load_dirs]
        args.labels = args.mc_names
        args.model_names = ["idepart", "ipart", "depart", "particle_net",
                            "part", "transformer", "efn", "pfn", "fc", "highway"]
        args.metric_names = ["gluon_rejection", "quark_efficiency", 'gluon_efficiency']
        args.reference = 'pythia'
        args.ref_dir = os.path.join(load_dir, 'Pythia8EvtGen_A14NNPDF23LO_jetjet')
        if wp == '50':
            args.ylims1 = [(5, 40), (0.1, 1.), (0.4, 1.)]
            args.ylims2 = [(-0.1, 0.175), (0.7, 1.3)]
        else:
            args.ylims1 = [(2, 7), (0.6, 1.), (0.4, 1.)]
            args.ylims2 = [(-0.1, 0.175), (0.7, 1.3)]
        args.xlog = False
        args.wp = f'{wp}%'
        main(args)

    # pt init dist comparison
    # args.labels = ['flat', 'log flat', 'weight flat']
    # args.load_dirs = ['/home/jankovys/JIDENN/logs/pythia_flat_70_JZ7/evaluation/pythia_phys',
    #                   '/home/jankovys/JIDENN/logs/pythia_log_flat_70_JZ7/evaluation/pythia_phys_2.5TeV',
    #                   '/home/jankovys/JIDENN/logs/pythia_W_flat_70_JZ10_noL/evaluation/pythia_phys']
    # args.model_names = ["transformer", "efn", "pfn", "fc", "highway"]
    # args.save_dir = '/home/jankovys/JIDENN/logs/compare_in_pt_dist'
    # args.metric_names = ["binary_accuracy", "quark_efficiency", 'gluon_efficiency']
    # args.reference = 'flat'
    # args.ref_dir = '/home/jankovys/JIDENN/logs/pythia_flat_70_JZ7/evaluation/pythia_phys'
    # args.ylims1 = None
    # args.ylims2 = None
    # args.xlog = False
