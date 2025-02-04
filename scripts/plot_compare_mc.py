import os
import sys
import numpy as np
sys.path.append(os.getcwd())
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import atlasify
import argparse
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
parser.add_argument("--release", default="r21", type=str,
                    help="MC type.")
parser.add_argument("--subtitle", default="", type=str,
                    help="Subtitle for the plots.")

# plt.rc('text', usetex=True)
# plt.rc('text.latex', preamble=r'\usepackage{amsmath}')


def main(args: argparse.Namespace):
    if args.release == 'r21':
        base_title = r'anti-$k_{\mathrm{T}}$, $R = 0.4$'
        base_title += '\n'
        base_title += args.subtitle
    elif args.release == 'r22':
        base_title = r'anti-$k_{\mathrm{T}}$, $R = 0.4$'
        base_title += '\n'
        base_title += args.subtitle 
    else:
        raise ValueError('Release not supported')
    for model in args.model_names:
        dfs = []
        for dir in args.load_dirs:
            df = pd.read_csv(os.path.join(
                dir, 'models', f'{model}', 'binned_metrics.csv'))
            df.to_csv(os.path.join(dir, 'models',
                      f'{model}', 'binned_metrics.csv'), index=False)
            if np.min(df['bin_mid'].values) < 20 and args.type == 'jets_pt':
                df['bin_mid'] = (df['bin_mid']*1e3).round(3)
                df['bin_width'] = (df['bin_width']*1e3).round(3)
            else:
                df['bin_mid'] = df['bin_mid'].round(3)
                df['bin_width'] = df['bin_width'].round(3)
            dfs.append(df)
        hm_labels = []
        for label in args.labels:
            hm_labels.append(MC_NAMING_SCHEMA[label] if label in MC_NAMING_SCHEMA else label)
        plot_var_dependence(dfs=dfs,
                            labels=hm_labels,
                            bin_midpoint_name='bin_mid',
                            bin_width_name='bin_width',
                            n_counts=args.metric_n_counts,
                            metric_names=args.metric_names,
                            save_path=os.path.join(
                                args.save_dir, 'models', f'{model}'),
                            ratio_reference_label=MC_NAMING_SCHEMA[args.reference] if args.reference in MC_NAMING_SCHEMA else args.reference,
                            xlabel=r'$p_T$ [GeV]' if args.type == 'jets_pt' else r'$\eta$',
                            ylabel_mapper=METRIC_NAMING_SCHEMA,
                            ylims=args.ylims1,
                            figsize=(8.6, 7),
                            xlog=args.xlog,
                            #h_line_position=[None, args.wp_val, None],
                            leg_loc='upper right',
                            title=f'{MODEL_NAMING_SCHEMA[model] if model in MODEL_NAMING_SCHEMA.keys() else model}, {args.wp} WP\n{base_title}',
                            colours=sns.color_palette(
                                "Set1", len(args.labels)),
                            label_fontsize=19,
                            fontsize=17,
                            leg_fontsize=17,
                            atlas_fontsize=17,
                            markersize=8,
                            )

    for metric in ['quark_efficiency', 'gluon_rejection', 'binary_accuracy', 'auc']:
        df_big = pd.DataFrame()
        # overall_metrics = pd.read_csv(args.overall_metrics, index_col=0)
        # acc_sorted_models = overall_metrics.sort_values(by='gluon_rejection_at_quark_80wp', ascending=False).index

        for label, dir in zip(args.labels, args.load_dirs):
            dfs = []
            for model in args.model_names:
                ref_df = pd.read_csv(os.path.join(
                    args.ref_dir, 'models', f'{model}', 'binned_metrics.csv'))
                if np.min(ref_df['bin_mid'].values) < 20 and args.type == 'jets_pt':
                    ref_df['bin_mid'] = (ref_df['bin_mid']*1e3).round(3)
                    ref_df['bin_width'] = (ref_df['bin_width']*1e3).round(3)
                else:
                    ref_df['bin_mid'] = ref_df['bin_mid'].round(3)
                    ref_df['bin_width'] = ref_df['bin_width'].round(3)
                if model == args.reference:
                    continue
                df = pd.read_csv(os.path.join(
                    dir, 'models', f'{model}', 'binned_metrics.csv'))
                if np.min(df['bin_mid'].values) < 20 and args.type == 'jets_pt':
                    df['bin_mid'] = (df['bin_mid']*1e3).round(3)
                    df['bin_width'] = (df['bin_width']*1e3).round(3)
                else:
                    df['bin_mid'] = df['bin_mid'].round(3)
                    df['bin_width'] = df['bin_width'].round(3)
                df['ratio'] = df[metric] / ref_df[metric]
                df['diff'] = ref_df[metric] - df[metric]
                dfs.append(df)
                df['mc'] = label
                df['model'] = model
                df_big = pd.concat([df_big, df])

        def calc_diff(df_big):
            df = pd.DataFrame()
            df['had_diff'] = (df_big[df_big['mc'] == 'sherpa'][metric] - df_big[df_big['mc'] ==
                                                                                'sherpa_lund'][metric]) / df_big[df_big['mc'] == 'pythia'][metric]
            df['ps_diff'] = (df_big[df_big['mc'] == 'herwig7'][metric] - df_big[df_big['mc'] ==
                             'herwig7_dipole'][metric]) / df_big[df_big['mc'] == 'pythia'][metric]
            df['ps_diff_sherpa'] = (df_big[df_big['mc'] == 'sherpa'][metric] - df_big[df_big['mc'] ==
                                                                                      'sherpa_dire'][metric]) / df_big[df_big['mc'] == 'pythia'][metric]
            return df

        df_en = df_big.copy()[['model', 'bin_mid',
                               'bin_width', 'ratio', 'diff']]
        df_en['ratio'] = (df_en['ratio'] - 1).abs()
        df_en['diff'] = df_en['diff'].abs()

        df_envel = df_en.groupby(
            ['model', 'bin_mid', 'bin_width']).mean().reset_index()
        df_envel = df_envel.rename(
            columns={'ratio': 'ratio_mean', 'diff': 'diff_mean'})

        df_envel2 = df_en.groupby(
            ['model', 'bin_mid', 'bin_width']).max().reset_index()
        df_envel2 = df_envel2.rename(
            columns={'ratio': 'ratio_max', 'diff': 'diff_max'})

        df_envel = df_envel.merge(
            df_envel2, on=['model', 'bin_mid', 'bin_width'])
        dfs_eval = []
        labels = []
        for model in df_envel['model'].unique():
            df = df_envel[df_envel['model'] == model]
            labels.append(model)
            dfs_eval.append(df)

        labels, dfs_eval = zip(
            *sorted(zip(labels, dfs_eval), key=lambda x: args.model_names.index(x[0])))
        new_labels = []
        for label in labels:
            new_labels.append(
                MODEL_NAMING_SCHEMA[label] if label in MODEL_NAMING_SCHEMA else label)
        labels = new_labels
        # model_label = r'\varepsilon^{-1}_{g,\mathrm{model}}' if metric == 'gluon_rejection' else r'\varepsilon_{q,\mathrm{model}}'
        # model_label = r'\varepsilon^{-1}_{g,\mathrm{model}}' if metric == 'gluon_rejection' else r'\varepsilon_{q,\mathrm{model}}'
        # py_label = r'\varepsilon^{-1}_{g,\mathrm{Pythia}}' if metric == 'gluon_rejection' else r'\varepsilon_{q,\mathrm{Pythia}}'
        if metric == 'quark_efficiency':
            model_label = r'\varepsilon_{q,\mathrm{model}}'
            py_label = r'\varepsilon_{q,\mathrm{Pythia}}'
        elif metric == 'gluon_rejection':
            model_label = r'\varepsilon^{-1}_{g,\mathrm{model}}'
            py_label = r'\varepsilon^{-1}_{g,\mathrm{Pythia}}'
        elif metric == 'binary_accuracy':
            model_label = r'\mathrm{ACC}_{\mathrm{model}}'
            py_label = r'\mathrm{ACC}_{\mathrm{Pythia}}'
        elif metric == 'auc':
            model_label = r'\mathrm{AUC}_{\mathrm{model}}'
            py_label = r'\mathrm{AUC}_{\mathrm{Pythia}}'
        else:
            raise ValueError('Metric not supported')
        
        # diff_label = r'$\varepsilon^{-1}_{g}$' if metric == 'gluon_rejection' else r'$\varepsilon_{q}$'
        plot_var_dependence(dfs=dfs_eval,
                            labels=labels,
                            bin_midpoint_name='bin_mid',
                            bin_width_name='bin_width',
                            metric_names=['ratio_mean', 'diff_mean', 'ratio_max',
                                          'diff_max'],
                            save_path=os.path.join(
                                args.save_dir, 'envelope', metric),
                            ratio_reference_label=None,
                            xlabel=r'$p_T$ [GeV]',
                            ylabel_mapper={'ratio_mean': r'mean $\mid 1 - ' + model_label + r'\ / \ ' + py_label + r' \mid $',
                                           'diff_mean': r'mean $\mid 1 - ' + model_label + r' - ' + py_label + r' \mid $',
                                           'ratio_max': r'max $\mid 1 - ' + model_label + r'\ / \ ' + py_label + r' \mid $',
                                           'diff_max': r'max $\mid 1 - ' + model_label + r' - ' + py_label + r' \mid $',
                                           },
                            # h_line_position=[None, None,
                            #                  None, None, 0.0, 0.0, 0.0],
                            xlog=args.xlog,
                            figsize=(11.5, 8.),
                            # ylims=[None, None, (0.03, 0.2), None, (-0.16, 0.16), (-0.16, 0.16), None] if metric == 'quark_efficiency' else [
                            #     None, None, (0.05, 0.8), None, (-0.25, 0.25), (-0.2, 0.7), None],
                            leg_loc='upper right',
                            title=[f'{args.wp} WP\n{base_title}', f' {args.wp} WP\n{base_title}', f' {args.wp} WP\n{base_title}', f' {args.wp} WP\n{base_title}'],
                            colours=sns.color_palette(
                                "colorblind", len(args.model_names)),)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # mc comparison
    # wp = '50'
    jet_type = 'UFO'
    args.subtitle = 'PFlow jets' if jet_type == 'PFO' else 'UFO CSSK jets' 
    opts = [('comparison_proc', ['Pythia8EvtGen_A14NNPDF23LO_jetjet',
                                    'PhPy8EG_A14_ttbar_hdamp258p75_nonallhad',
                                    'PowhegPy8EG_NNLOPS_nnlo_30_ggH125_tautaul13l7',
                                    'PowhegPythia8EvtGen_AZNLOCTEQ6L1_Zmumu',
                                    # 'PowhegPy8EG_NNPDF30_AZNLOCTEQ6L1_VBFH125_tautaul13l7',
                                    'Sherpa_222_NNPDF30NNLO_SinglePhoton']),
            ('comparison_mc', ['Pythia8EvtGen_A14NNPDF23LO_jetjet',
                                'PhPy8EG_jj',
                                'H7EG_Matchbox_angular_jetjetNLO',
                                'H7EG_Matchbox_dipole_jetjetNLO',
                                # 'PowhegPy8EG_NNPDF30_AZNLOCTEQ6L1_VBFH125_tautaul13l7',
                                'Sherpa_CT10_CT14nnlo_CSShower_2to2jets',
                                'Sherpa_CT10_CT14nnlo_CSShower_Lund_2to2jets'])]
    # opts = opts[1:]
    
    for  save_dir_name, load_dirs in opts:
        for var in ['jets_pt', 'jets_eta']:
            args.type = var
            for wp_cfg in ['quark_50wp/']:
                for wp_val in [0.5]:
                    print(wp_val)
                    args.wp_val = wp_val
                    wp = str(int(wp_val * 100))
                    args.wp = wp
                    args.release = 'r22'
                    load_dir = f'logs/r22_UFO_2Dflat_200GeV/evaluation/central_2d_flat_20-200GeV/{wp_cfg}{var}'
                    
                    args.load_dirs = load_dirs
                    args.save_dir = os.path.join(load_dir, save_dir_name)
                    
                    
                    args.mc_names = [MC_NAMING_SCHEMA[mc] for mc in args.load_dirs]
                    args.load_dirs = [os.path.join(load_dir, mc) for mc in args.load_dirs]
                    args.labels = args.mc_names

                    # args.model_names = ["idepart", "idepart+topo", "fc-no-eta", "fc_crafted-no_eta",
                    #                     "pfn", "pfn+topo", "fc_crafted", 'efn', 'efn+topo']
                    # args.model_names = ["idepart-rel", "highway", "pfn-rel", "efn", "fc_crafted"]
                    # args.model_names = ["idepart-rel-allMC", "highway-allMC", "pfn-rel-allMC", "efn-allMC", "fc_crafted-allMC"]
                    args.model_names = ["idepart", "idepart-rel",  "idepart-finetunne", "pfn", "fc_crafted", "efn"]
                    # args.model_names = ["idepart", "idepart-rel", "fc_crafted"]
                    # args.model_names = ["idepart-rel", "idepart-rel-finetune", "idepart-rel-finetune-2"]
                    # args.model_names = ["idepart-rel", "idepart-rel-finetune", "idepart-rel-low", "idepart-rel-mid", "fc_crafted", 'highway', 'efn', 'pfn-rel']
                    # args.model_names = ["idepart-rel", "idepart-rel-finetune", "idepart-rel-finetune-2", "idepart-rel-finetune-3", "idepart-rel-finetune-5", "idepart-rel-finetune-4"]

                    args.metric_names = ["gluon_rejection", "quark_rejection", "binary_accuracy",
                                        "quark_efficiency", 'gluon_efficiency', 'auc', 'gluon_rejection_at_quark_50wp']
                    args.metric_n_counts = ['eff_num_events_g', "eff_num_events_q", "eff_num_events",
                                            'eff_num_events_q', 'eff_num_events_g', 'eff_num_events', 'eff_num_events_q']
                    # args.reference = None
                    # args.ref_dir = None
                    args.reference = args.mc_names[0]
                    args.ref_dir = os.path.join(load_dir,'Pythia8EvtGen_A14NNPDF23LO_jetjet')
                    args.ylims1 = None
                    args.ylims2 = None
                    # if save_dir_name == 'comparison_proc+vbf' or save_dir_name == 'comparison_proc_vbf+vbf':
                    #     if wp_cfg == 'quark_50wp/' and var == 'jets_pt':
                    #         args.ylims1 = [(2, 30), None, None, (0.33, 0.70), None, None, None]
                    #     if wp_cfg == 'quark_50wp/' and var == 'jets_eta':
                    #         args.ylims1 = [(3, 18), None, None, (0.33, 0.65), None, None, None]
                    # else:
                    #     if wp_cfg == 'quark_50wp/' and var == 'jets_pt':
                    #         args.ylims1 = [(3, 14.5), None, None, (0.42, 0.55), None, None, None]
                    #     if wp_cfg == 'quark_50wp/' and var == 'jets_eta':
                    #         args.ylims1 = [(4.8, 11), None, None, (0.4, 0.56), None, None, None]

                    args.xlog = False
                    args.wp = f'{wp}%'
                    try:
                        main(args)
                    except Exception as e:
                        print(e)
                        continue
            
            

