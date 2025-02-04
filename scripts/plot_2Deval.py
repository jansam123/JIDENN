import os
import sys
sys.path.append(os.getcwd())
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import atlasify
import puma
import argparse

from sklearn.metrics import roc_auc_score, accuracy_score
from jidenn.const import METRIC_NAMING_SCHEMA, LATEX_NAMING_CONVENTION, MODEL_NAMING_SCHEMA
from jidenn.evaluation.evaluation_metrics import EfficiencyAtFixedWorkingPoint


parser = argparse.ArgumentParser()
parser.add_argument("--load_path", type=str, help="Path to the .csv file")
parser.add_argument("--save_path", type=str, help="Path to save the plots")
parser.add_argument("--second_bin_var", type=str, help="Second bin variable. Options: eta, mu")

THRESHOLD = 0.5

def main(args):

    def calc_metrics(df):
        
        metric_10wp = EfficiencyAtFixedWorkingPoint(name='gluon_efficiency_at_quark_10wp', fixed_label_id=1, working_point=0.1, returned_label_id=0)
        metric_20wp = EfficiencyAtFixedWorkingPoint(name='gluon_efficiency_at_quark_20wp', fixed_label_id=1, working_point=0.2, returned_label_id=0)
        metric_30wp = EfficiencyAtFixedWorkingPoint(name='gluon_efficiency_at_quark_30wp', fixed_label_id=1, working_point=0.3, returned_label_id=0)
        metric_40wp = EfficiencyAtFixedWorkingPoint(name='gluon_efficiency_at_quark_40wp', fixed_label_id=1, working_point=0.4, returned_label_id=0)
        metric_50wp = EfficiencyAtFixedWorkingPoint(name='gluon_efficiency_at_quark_50wp', fixed_label_id=1, working_point=0.5, returned_label_id=0)
        metric_60wp = EfficiencyAtFixedWorkingPoint(name='gluon_efficiency_at_quark_60wp', fixed_label_id=1, working_point=0.6, returned_label_id=0)
        metric_70wp = EfficiencyAtFixedWorkingPoint(name='gluon_efficiency_at_quark_70wp', fixed_label_id=1, working_point=0.7, returned_label_id=0)
        metric_80wp = EfficiencyAtFixedWorkingPoint(name='gluon_efficiency_at_quark_80wp', fixed_label_id=1, working_point=0.8, returned_label_id=0)
        metric_90wp = EfficiencyAtFixedWorkingPoint(name='gluon_efficiency_at_quark_90wp', fixed_label_id=1, working_point=0.9, returned_label_id=0)
        
        
        metrics = [metric_10wp, metric_20wp, metric_30wp, metric_40wp, metric_50wp, metric_60wp, metric_70wp, metric_80wp, metric_90wp]
        names = ['0.1wp', '0.2wp', '0.3wp', '0.4wp', '0.5wp', '0.6wp', '0.7wp', '0.8wp', '0.9wp']
        # names = [f'gluon_rejection_at_{wp}_quark_efficiency' for wp in wps]
        eff_dict = {name: 1 - metric(y_true=df['label'].values, y_pred=df['score'].values, sample_weight=df['weight'].values).numpy() for metric, name in zip(metrics, names)}
        # for metric, metrics_name in zip(metrics, names):
        #     df[metrics_name] = metric(y_true=df['label'].values, y_pred=df['score'].values, sample_weight=df['weight'].values).numpy()
        
        
        # accuracy = accuracy_score(df['label'].values, df['pred'].values, sample_weight=df['weight'].values)
        # N_eff = np.sum(df['weight'].values)**2 / np.sum(df['weight'].values**2)
        # accuracy_err = np.sqrt(accuracy * (1 - accuracy) / N_eff)
        # auc = roc_auc_score(df['label'].values, df['score'].values, sample_weight=df['weight'].values)
        # auc_err = np.sqrt(auc * (1 - auc) / N_eff)     
        # q_eff = np.sum(q_df[q_df['pred'] == 1]['weight'].values) / np.sum(q_df['weight'].values)
        # g_eff = np.sum(g_df[g_df['pred'] == 0]['weight'].values) / np.sum(g_df['weight'].values)
        # q_df = df[df['label'] == 1]
        # N_q = np.sum(q_df['weight'].values)**2 / np.sum(q_df['weight'].values**2)
        g_df = df[df['label'] == 0]
        N_g = np.sum(g_df['weight'].values)**2 / np.sum(g_df['weight'].values**2)
        eff_err_dict = {name+'_err': np.sqrt(eff * (1 - eff) / N_g) for name, eff in eff_dict.items()} 
        
        # return pd.Series({'accuracy': accuracy, 'auc': auc, 'accuracy_err': accuracy_err, 'auc_err': auc_err, 'q_eff': q_eff, 'g_eff': g_eff, 'N_q': N_q, 'N_g': N_g})
        return pd.Series({**eff_dict, **eff_err_dict})
    
    # def str_to_eta(x):
    #     x = x.split(']')[0].replace('(', '').replace(' ', '').split(',')
    #     return (float(x[0]) + float(x[1])) / 2
    
    # def str_to_pt(x):
    #     x = x.split(']')[1].replace('(', '').replace(' ', '').split(',')
    #     return (float(x[0]) + float(x[1])) / 2
    
    save_path = os.path.dirname(args.load_path)
    save_path = os.path.join(save_path, '2D_metric_plots')
    os.makedirs(os.path.join(save_path, 'png'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'pdf'), exist_ok=True)
    
    df = pd.read_csv(args.load_path, index_col=0)
    # df = df.head(1000000)
    df = df.rename(columns={'idepart-rel_score': 'score'})
    print(df)
    # score = [col for col in df.columns if '_score' in col][0]
    # score = score.replace('_score', '')
    # df['score'] = df[score + '_score']
    df = df[['jets_pt', 'jets_eta', 'label', 'weight', 'score', 'corrected_averageInteractionsPerCrossing']].rename(columns={'corrected_averageInteractionsPerCrossing': 'mu'})
    # df['pred'] = np.where(df['score'] > THRESHOLD, 1, 0)
    df['jets_absEta'] = df['jets_eta'].abs()
    # df['jets_pt'] = df['jets_pt'] * 1e-3
    
    df['eta_bin'] = pd.cut(df['jets_absEta'], bins=[0, 0.5, 1.0, 1.5, 2.1, 3.6, 4.5])
    df['mu_bin'] = pd.cut(df['mu'], bins=[0, 20, 40, 60, 80, 100])
    df['pt_bin'] = pd.cut(df['jets_pt'], bins=[20_000, 50_000, 100_000, 200_000, 300_000, 500_000, 800_000, 1_000_000, 1_200_000, 1_500_000, 2_000_000, 2_500_000])
    # set index the bins
    if args.second_bin_var == 'eta':
        df = df.set_index(keys=['pt_bin', 'eta_bin'])
    elif args.second_bin_var == 'mu':
        df = df.set_index(keys=['pt_bin', 'mu_bin'])
    print(df)
    
    # df['eta_pt_bin'] = df['eta_bin'].astype(str) + df['pt_bin'].astype(str)
    df = df.groupby(axis=0, level=[0, 1], observed=False).apply(calc_metrics)
    # df['eta'] = df['eta_pt_bin'].apply(str_to_eta)
    # df['pt'] = df['eta_pt_bin'].apply(str_to_pt).astype(int)
    # df_to_save = df.rename(columns={'pt_bin': 'jets_pt [MeV]', 'eta_bin': 'jets_absEta'})
    # rename the index names
    df_to_save = df.rename(columns={f'{wp}wp':f'gluon_efficiency_at_{wp}_quark_efficiency' for wp in ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']})
    df_to_save.index.names = ['jets_pt [MeV]', 'jets_absEta'] if args.second_bin_var == 'eta' else ['jets_pt [MeV]', 'mu']
    df_to_save = df_to_save.rename(columns={f'{wp}wp_err':f'gluon_efficiency_at_{wp}_quark_efficiency_stat_err' for wp in ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']})
    df_to_save.to_csv(os.path.join(args.save_path, 'qg_tag_performance.csv'), na_rep=-1)
    err_df = df[[col for col in df.columns if 'err' in col]]
    df = df[[col for col in df.columns if 'err' not in col]]
    print(df)
    print(err_df)
    # labels = ['Accuracy', 'AUC', 'Accuracy Error', 'AUC Error', 'Quark Efficiency', 'Gluon Efficiency', '$N$ quarks', '$N$ gluons']
    # metrics = ['accuracy', 'auc', 'accuracy_err', 'auc_err', 'q_eff', 'g_eff', 'N_q', 'N_g']
    for wp_label in ['0.1wp', '0.2wp', '0.3wp', '0.4wp', '0.5wp', '0.6wp', '0.7wp', '0.8wp', '0.9wp']:
        print(wp_label)
        sm_df = df[[wp_label]].reset_index()
        err_sm_df = err_df[[wp_label+'_err']].reset_index()
        err_sm_df['pt_bin'] = err_sm_df['pt_bin'].map(lambda x: int(x.mid * 1e-3))
        sm_df['pt_bin'] = sm_df['pt_bin'].map(lambda x: int(x.mid * 1e-3))
        if args.second_bin_var == 'eta':
            sm_df['eta_bin'] = sm_df['eta_bin'].map(lambda x: x.mid)
            err_sm_df['eta_bin'] = err_sm_df['eta_bin'].map(lambda x: x.mid)
        elif args.second_bin_var == 'mu':
            sm_df['mu_bin'] = sm_df['mu_bin'].map(lambda x: x.mid)
            err_sm_df['mu_bin'] = err_sm_df['mu_bin'].map(lambda x: x.mid)
        sm_df = sm_df.pivot(index='pt_bin', columns='eta_bin' if args.second_bin_var == 'eta' else 'mu_bin', values=wp_label)
        sm_df = sm_df.iloc[::-1]
        err_sm_df = err_sm_df.pivot(index='pt_bin', columns='eta_bin' if args.second_bin_var == 'eta' else 'mu_bin', values=wp_label+'_err')
        err_sm_df = err_sm_df.iloc[::-1]
        sm_df[err_sm_df.isna()] = np.nan
        annot = sm_df.round(3).astype(str) + r' $\pm$ ' + err_sm_df.round(3).astype(str)
        vmin = 0.
        vmax = 1.
        fig = plt.figure(figsize=(10, 6))
        ax = sns.heatmap(sm_df, cmap='rocket_r', cbar_kws={'label': f'Gluon Efficiency at {wp_label.replace("wp","")} Quark Efficiency'}, vmin=vmin, vmax=vmax, annot=annot, fmt='')
        mask = sm_df.isna().values
        errors = err_sm_df.values
        values = sm_df.values
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if not mask[i, j]:
                    color = ax.collections[0].cmap(ax.collections[0].norm(values[i, j]))
                    alpha = min(1, errors[i, j] * 10)
                    rect = plt.Rectangle((j, i), 1, 1, fill=True, color=color, alpha=alpha)
                    ax.add_patch(rect)
                    
        plt.ylabel(r'$p_{\mathrm{T}}$ [GeV]', horizontalalignment='right', x=1.0, fontsize=12)
        if args.second_bin_var == 'eta':
            plt.xlabel(r'$|\eta|$', horizontalalignment='right', x=1.0, fontsize=12)
        elif args.second_bin_var == 'mu':
            plt.xlabel(r'$\mu$', horizontalalignment='right', x=1.0, fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        plt.yticks(rotation=0)
        

        atlasify.atlasify(
            axes=ax,
            outside=True,
            subtext=f'13 TeV, Pythia8, ' + r'anti-$k_{\mathrm{T}}$, $R = 0.4$ UFO CSSK jets, ' + f'{int(float(wp_label.replace("wp",""))*100)}% WP',
            atlas='Simulation Classified',
            font_size=12,
            sub_font_size=12,
            label_font_size=12,
        )
        plt.savefig(f"{args.save_path}/{wp_label}.png", bbox_inches='tight', dpi=400)
        plt.savefig(f"{args.save_path}/{wp_label}.pdf", bbox_inches='tight')
        plt.close()
            

# def main2(args):
#     import uproot
#     df = pd.read_csv(os.path.join(args.save_path, 'qg_tag_performance.csv'), index_col=[0,1])
#     err_df = df[['gluon_efficiency_at_0.5_quark_efficiency_stat_err']]
#     df = df[['gluon_efficiency_at_0.5_quark_efficiency']]
#     # switch order of multiindex
#     df = df.swaplevel()
#     data = np.reshape(df.values, -1)
#     data = np.concatenate([np.zeros_like(data), data])
#     err = np.reshape(err_df.values, -1)
#     sumw2 = err**2
#     sumw2 = np.concatenate([np.zeros_like(sumw2), sumw2])
#     print(len(data))
#     newfile = os.path.join(args.save_path, "qg_tag_performance.root")
    
#     h1 = uproot.writing.identify.to_TH2x(
#         fName="h1",
#         fTitle="Quark Efficiency at Fixed Gluon Efficiency",
#         data=data,
#         fEntries=2,
#         fTsumw=0.,
#         fTsumw2=0.,
#         fTsumwx=0.,
#         fTsumwx2=0.,
#         fTsumwy=0.,
#         fTsumwy2=0.,
#         fTsumwxy=0.,
#         fSumw2=sumw2,
#         fXaxis=uproot.writing.identify.to_TAxis(
#             fName="jet_eta",
#             fTitle="jet_eta",
#             fXbins=np.array([0, 1.0, 2.1, 3.6, 4.5]),
#             fNbins=4,
#             fXmin=0.,
#             fXmax=4.5,
            
#         ),
#         fYaxis=uproot.writing.identify.to_TAxis(
#             fName="jet_pt",
#             fTitle="jet_pt",
#             fXbins=np.array([20_000., 50_000., 100_000., 200_000., 300_000., 500_000., 800_000., 1_000_000., 1_200_000., 1_500_000., 2_000_000., 2_500_000.]),
#             fNbins=11,
#             fXmin=20_000,
#             fXmax=2_500_000,
#         ),
#     )
    
#     with uproot.recreate(newfile) as fout:
#         fout["out"] = h1



if __name__ == "__main__":
    args  = parser.parse_args()
    # args.load_path = 'logs/r22_PFO_piecewise_2Dflat_new/evaluation/fwd_phys_20-2500GeV/jets_pt/Pythia8EvtGen_A14NNPDF23LO_jetjet/models/idepart-rel/score_dataset.csv'
    # args.save_path = 'logs/r22_PFO_piecewise_2Dflat_new/evaluation/fwd_phys_20-2500GeV/jets_pt/Pythia8EvtGen_A14NNPDF23LO_jetjet/2D_plots'
    main(args)
    # main2(args)
