import os
import sys
sys.path.append(os.getcwd())
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np
from io import BytesIO
from typing import List, Union, Dict, Optional, Tuple
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
parser.add_argument("--filename", type=str, help="Name of csv file containing binned metrics")
parser.add_argument("-m", "--model_names", nargs='*', type=str,
                    help="names of the models.")
parser.add_argument("--save_dir", default=".", type=str,
                    help="Directory to save the plots to.")
parser.add_argument("--var", default="jets_pt", type=str,
                    help="Variable to plot.")
parser.add_argument("--type", default="pT", type=str, help="Type of the plot.")
parser.add_argument("--compare_mc", default=False, type=bool,
                    help="Compare MC models.")
parser.add_argument("-f", "--figsize", default=(14, 9), type=float, nargs=2,  # (13, 9)
                    help="Figure size.")
parser.add_argument("--leg_ncol", default=1, type=int,
                    help="Number of columns in the legend.")
parser.add_argument("--mc_type", default="Pythia8EvtGen_A14NNPDF23LO_jetjet", type=str,
                    help="MC type.")
parser.add_argument("--release", default="r22", type=str,
                    help="MC type.")
parser.add_argument("--subtitle", default="", type=str,
                    help="Subtitle for the plots.")
parser.add_argument("--ylim", default=None, type=float, nargs=2,
                    help="Y-axis limits.")


def compare_ml_models(paths: List[str],
                      labels: List[str],
                      save_path: str,
                      figsize: Tuple[int, int] = (12, 8),
                      leg_ncol: int = 2,
                      x_var='jets_pt',
                      sublabel: str | None = None):

    os.makedirs(save_path, exist_ok=True)
    dfs = [pd.read_csv(path) for path in paths]
    accuracies = [df['gluon_rejection_at_quark_50wp'].mean() for df in dfs]
    # labels, dfs, accuracies = zip(*sorted(zip(labels, dfs, accuracies),
    #                                                     key=lambda x: x[2], reverse=True))
    for label, acc in zip(labels, accuracies):
        print(f'{label}: {acc:.4f}')
    pd.DataFrame({'Model': labels, 'Accuracy': accuracies}).to_csv(
        os.path.join(save_path, 'sorted_accuracies.csv'), index=False)
    print(labels)
    metric_names = ["gluon_efficiency", "quark_efficiency",
                    "gluon_rejection", "quark_rejection",
                    "binary_accuracy", "auc",
                    # "effective_tagging_efficiency",
                    # 'gluon_rejection_at_quark_80wp', 'gluon_efficiency_at_quark_80wp',
                    'gluon_rejection_at_quark_50wp', #'gluon_efficiency_at_quark_50wp',
                    # 'quark_efficiency_at_gluon_80wp',
                    # 'quark_efficiency_at_gluon_50wp',
                    ]
    n_counts = ['eff_num_events_g', 'eff_num_events_q',
                'eff_num_events_g', 'eff_num_events_q',
                'eff_num_events', 'eff_num_events',
                # 'eff_num_events_g',
                # 'eff_num_events_g', 'eff_num_events_g',
                'eff_num_events_g', #'eff_num_events_g',
                # 'eff_num_events_q',
                # 'eff_num_events_q',
                ]

    # ylims = [[0.6, 0.9], [0.55, 0.9], [0.6, 0.85], [0.7, 0.9], [0.2, 0.5],
    #          [2.5, 6], [3, 35], [0.53, 0.85], [0.85, 1.0]]
    # if x_var == 'jets_pt':
    #     ylims = [None]*6 + [[2., 6.8], None, [6.5, 36.], None]
    # elif x_var == 'jets_eta':
    #     ylims = [None]*6 + [[1., 4.8], None, [3., 22.5], None]
    # ylims=None
    colours = sns.color_palette('colorblind', len(labels))
    # colours = sns.color_palette('Paired', len(labels))
    # if x_var == 'jets_pt':
    #     for df in dfs:
    #         df['bin_mid'] = df['bin_mid'] * 1e-6
    #         df['bin_width'] = df['bin_width'] * 1e-6
    if x_var == 'jets_eta':
        for df in dfs:
            df = df[df['bin_mid'] > 0]
            
    if x_var == 'jets_pt':
         for df in dfs:
                if np.min(df['bin_mid'].values) < 1:
                    df['bin_mid'] = df['bin_mid']*1e3
                    df['bin_width'] = df['bin_width']*1e3
    
    title_50 = ', 50% WP\n' if sublabel is not None or sublabel != '' else '50% WP\n'
    # if args.release == 'r21':
    #     title_all = r'anti-$k_{\mathrm{T}}$, $R = 0.4$ PFlow jets'
    #     title_all += '\n'
    # elif args.release == 'r22':
    #     title_all += '\n'
    # else:
    #     raise ValueError('Release not supported')
    title_all = r'anti-$k_{\mathrm{T}}$, $R = 0.4$' + '\n' 
    title_all += sublabel if sublabel else ''
    new_labels = []
    for label in labels:
        new_labels.append(
            MODEL_NAMING_SCHEMA[label] if label in MODEL_NAMING_SCHEMA else label)
    plot_var_dependence(dfs=dfs,
                        labels=new_labels,
                        bin_midpoint_name='bin_mid',
                        bin_width_name='bin_width',
                        n_counts=n_counts,
                        metric_names=metric_names,
                        save_path=save_path,
                        ratio_reference_label=None,
                        xlabel=LATEX_NAMING_CONVENTION[x_var] if x_var in LATEX_NAMING_CONVENTION else x_var,
                        ylabel_mapper=METRIC_NAMING_SCHEMA,
                        ylims=args.ylims,
                        xlims=None,
                        xlog=False,
                        # title=[title_none + title_all]*6 + [title_80 +
                        #                                     title_all]*2 + [title_50 + title_all]*2 + [title_80 +
                        #                                     title_all]*1 + [title_50 + title_all]*1,
                        title=[title_all]*6 + [title_all+title_50],
                        figsize=figsize,
                        leg_loc='upper right',
                        leg_ncol=leg_ncol,
                        colours=colours,
                        label_fontsize=30,
                        fontsize=24,
                        leg_fontsize=24,
                        atlas_fontsize=24,
                        markersize=12
                        )


def main(args: argparse.Namespace):

    paths = []
    for model in args.model_names:
        if not os.path.exists(os.path.join(args.load_dir, model, 'binned_metrics', args.filename)):
            print(f'No binned metrics for {model} in {os.path.join(args.load_dir, model, "binned_metrics", args.filename)}')
            continue
        paths += [os.path.join(args.load_dir, model, 'binned_metrics', args.filename)]

    compare_ml_models(paths=paths,
                      labels=args.model_names,
                      save_path=os.path.join(args.save_dir, args.filename.rstrip('.csv')),
                      figsize=(args.figsize[0], args.figsize[1]),
                      leg_ncol=args.leg_ncol,
                      x_var=args.var,
                      sublabel=args.subtitle)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    
    # args.model_names = ["idepart-rel-2jets",  "idepart-rel-allMC", "pfn-rel",  "highway-allMC",  "highway", "fc_crafted", "efn"]
    # args.model_names = ["idepart", "idepart-rel",  "idepart-finetunne", "pfn", "fc_crafted", "efn"]
    # args.model_names = ["idepart-rel-UFO", "idepart-rel-PFO", "highway-UFO", "highway-PFO", "pfn-rel-UFO", "pfn-rel-PFO"]
    # args.model_names = ["idepart-rel", "idepart-rel-noTopo-v1"]
    # args.model_names = ["idepart-rel"]
    # args.model_names = ["highway", "highway-old"]
    # args.model_names = ["idepart-rel-allj", "idepart-rel-80const"]
    args.model_names = ["idepart-rel-allj", "idepart-rel_FPA", "pfn-rel"]
    # args.subtitle = 'UFO CSSK jets' 
    common_subtitle = 'PFlow jets' 
    data = 'Pythia8EvtGen_A14NNPDF23LO_jetjet' 
    mc_label = MC_NAMING_SCHEMA[data]
    common_subtitle += f', {mc_label}\n'
    # data = 'PowhegPy8EG_NNPDF30_AZNLOCTEQ6L1_VBFH125_tautaul13l7' 
    print(f'Processing {data}...')
    # for eval_type in ['central_2d_flat_20-200GeV', 'central_2d_flat_160-1300GeV', 'fwd_2d_flat_20-160GeV', 'super_central_2d_flat_1300-2500GeV']: #, 'fwd_phys_20-2500GeV']:
    variables = ['jets_pt', 'jets_eta', 'jets_eta', 'jets_eta', 'jets_eta', 'jets_eta']
    files = ['jets_pt.csv', 'jets_eta.csv', 'jets_eta_20-60.csv', 'jets_eta_60-160.csv', 'jets_eta_160-500.csv', 'jets_eta_500+.csv'] 
    subtitles = ['', '', r'$p_\mathrm{T} \in \left[ 20; 60 \right] $ GeV', r'$p_\mathrm{T} \in \left[ 60; 160 \right] $ GeV', r'$p_\mathrm{T} \in \left[ 160; 500 \right] $ GeV', r'$p_\mathrm{T} > 500$sq GeV']
    variables += ['jets_Constituent_n', 'jets_TopoTower_n', 'jets_Constituent_n+jets_TopoTower_n', 'jets_index', 'jets_index', 'jets_index', 'jets_index', 'jets_index']
    files += ['jets_Constituent_n.csv', 'jets_TopoTower_n.csv', 'jets_Constituent_n+jets_TopoTower_n.csv', 'jets_index.csv', 'jets_index_20-60.csv', 'jets_index_60-160.csv', 'jets_index_160-500.csv', 'jets_index_500+.csv']
    subtitles += ['', '', '', '', r'$p_\mathrm{T} \in \left[ 20; 60 \right] $ GeV', r'$p_\mathrm{T} \in \left[ 60; 160 \right] $ GeV', r'$p_\mathrm{T} \in \left[ 160; 500 \right] $ GeV', r'$p_\mathrm{T} > 500$ GeV']
    
    variables = ['jets_pt']
    files = ['jets_pt.csv'] 
    subtitles = ['',]
    for eval_type in ['fwd_phys_20-2500GeV']:
        print(f'Eval type: {eval_type}')
        for var, filename, subtitle in zip(variables, files, subtitles):#, 'corrected_averageInteractionsPerCrossing']:
        # for var in ['corrected_averageInteractionsPerCrossing']:
            print(f'Processing {var}...')
            args.load_dir = f'logs/r22_PFO_piecewise_2Dflat_v2/evaluation/{eval_type}/{data}/models'
            # args.save_dir = f'logs/r22_PFO_piecewise_2Dflat_v2/evaluation/{eval_type}/{data}/plots_80const/{var}'
            args.save_dir = f'logs/r22_PFO_piecewise_2Dflat_v2/evaluation/{eval_type}/{data}/plots_FPA/{var}'
            # args.load_dir = f'logs/eval_test'
            # args.save_dir = f'logs/plots_test'
            args.filename = filename
            args.mc_type = data
            args.var = var
            args.subtitle = common_subtitle + subtitle
            args.ylims = None #[None, None, None, None,None, None,(3, 14)] if var == 'jets_pt' else  [None, None, None, None,None, None,(4.8, 10.2)]
            # args.model_names = ["idepart", "idepart-rel", "fc_crafted"]
            # args.model_names = ["idepart", "idepart+topo", "idepart+topo+track", "idepart-rel",  "idepart-rel+topo",  "idepart-rel+topo+track",]
            try:
                main(args)
            except Exception as e:
                print(e)

    # for data in ['Pythia8EvtGen_A14NNPDF23LO_jetjet', 
    #              'PhPy8EG_A14_ttbar_hdamp258p75_nonallhad', 
    #              'PowhegPy8EG_NNLOPS_nnlo_30_ggH125_tautaul13l7', 
    #              'PowhegPythia8EvtGen_AZNLOCTEQ6L1_Zmumu', 
    #              'Sherpa_222_NNPDF30NNLO_SinglePhoton']:
    # for log_type in ['fwd_1d_phys_2500GeV-central','fwd_1d_phys_2500GeV-from200GeV']:
    # args.model_names = ["idepart", "idepart-rel", "pfn", "fc_crafted", 'efn']
    # for jet_type in ['PFO', 'UFO']:
    #     print(f'Processing {jet_type}...')
    #     args.subtitle = 'PFlow jets' if jet_type == 'PFO' else 'UFO CSSK jets' 
    #     for data in ['Pythia8EvtGen_A14NNPDF23LO_jetjet',]: 
    #     # for data in ['Pythia8EvtGen_A14NNPDF23LO_jetjet', 
    #     #              'H7EG_Matchbox_angular_jetjetNLO',
    #     #              'H7EG_Matchbox_dipole_jetjetNLO',
    #     #              'Sherpa_CT10_CT14nnlo_CSShower_2to2jets',
    #     #              'PhPy8EG_jj',
    #     #              'PhPy8EG_A14_ttbar_hdamp258p75_nonallhad', 
    #     #              'PowhegPy8EG_NNPDF30_AZNLOCTEQ6L1_VBFH125_tautaul13l7',
    #     #              'PowhegPy8EG_NNLOPS_nnlo_30_ggH125_tautaul13l7', 
    #     #              'PowhegPythia8EvtGen_AZNLOCTEQ6L1_Zmumu', 
    #     #              'Sherpa_222_NNPDF30NNLO_SinglePhoton']:
    #         print(f'Processing {data}...')
    #         for var in ['jets_pt', 'jets_eta']:
    #         # for var in ['jets_pt']:
    #             print(f'Processing {var}...')
    #             args.load_dir = f'logs/r22_{jet_type}_2Dflat_200GeV/evaluation/central_2d_flat_200GeV/{var}/{data}/models'
    #             args.save_dir = f'logs/r22_{jet_type}_2Dflat_200GeV/evaluation/central_2d_flat_200GeV/{var}/{data}/plots'
    #             args.mc_type = data
    #             args.var = var
    #             # args.ylims = [None, None, None, None,None, None,(3, 14)] if var == 'jets_pt' else  [None, None, None, None,None, None,(4.8, 10.2)]
    #             # args.model_names = ["idepart", "idepart-rel", "fc_crafted"]
    #             # args.model_names = ["idepart", "idepart+topo", "idepart+topo+track", "idepart-rel",  "idepart-rel+topo",  "idepart-rel+topo+track",]
    #             try:
    #                 main(args)
    #             except Exception as e:
    #                 print(e)
                    
        # for data in ['Pythia8EvtGen_A14NNPDF23LO_jetjet', 
        #             'H7EG_Matchbox_angular_jetjetNLO',
        #             'H7EG_Matchbox_dipole_jetjetNLO',
        #             'Sherpa_CT10_CT14nnlo_CSShower_2to2jets',
        #             'PhPy8EG_jj',
        #             'PhPy8EG_A14_ttbar_hdamp258p75_nonallhad', 
        #             'PowhegPy8EG_NNPDF30_AZNLOCTEQ6L1_VBFH125_tautaul13l7',
        #             'PowhegPy8EG_NNLOPS_nnlo_30_ggH125_tautaul13l7', 
        #             'PowhegPythia8EvtGen_AZNLOCTEQ6L1_Zmumu', 
        #             'Sherpa_222_NNPDF30NNLO_SinglePhoton']:
            
        #     print(f'Processing {data}...')
        #     for wp in [ 'quark_50wp']:
        #         print(f'Processing {wp}...')
        #         for var in ['jets_pt', 'jets_eta']:
        #             print(f'Processing {var}...')
        #             args.ylims = None
        #             args.load_dir = f'logs/r22_{jet_type}_2Dflat_200GeV/evaluation/central_2d_flat_200GeV/{wp}/{var}/{data}/models'
        #             args.save_dir = f'logs/r22_{jet_type}_2Dflat_200GeV/evaluation/central_2d_flat_200GeV/{wp}/{var}/{data}/plots'
        #             args.mc_type = data
        #             args.var = var
        #             try:
        #                 main(args)
        #             except Exception as e:
        #                 print(e)
                

                    
