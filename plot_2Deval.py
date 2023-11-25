import seaborn as sns
import matplotlib.pyplot as plt
from puma import Histogram, HistogramPlot
import pandas as pd
import numpy as np
import os
import atlasify
import puma
import argparse
from sklearn.metrics import roc_auc_score, accuracy_score
from jidenn.const import METRIC_NAMING_SCHEMA, LATEX_NAMING_CONVENTION, MODEL_NAMING_SCHEMA
from jidenn.evaluation.evaluation_metrics import RejectionAtFixedWorkingPoint


parser = argparse.ArgumentParser()
parser.add_argument("--load_path", type=str, help="Path to the .csv file")

THRESHOLD = 0.5

def calc_accuracy(df):
    accuracy = accuracy_score(df['label'].values, df['pred'].values, sample_weight=df['weight'].values)
    N_eff = np.sum(df['weight'].values)**2 / np.sum(df['weight'].values**2)
    accuracy_err = np.sqrt(accuracy * (1 - accuracy) / N_eff)
    auc = roc_auc_score(df['label'].values, df['score'].values, sample_weight=df['weight'].values)
    auc_err = np.sqrt(auc * (1 - auc) / N_eff)     
    q_df = df[df['label'] == 1]
    g_df = df[df['label'] == 0]
    q_eff = np.sum(q_df[q_df['pred'] == 1]['weight'].values) / np.sum(q_df['weight'].values)
    g_eff = np.sum(g_df[g_df['pred'] == 0]['weight'].values) / np.sum(g_df['weight'].values)
    N_q = np.sum(q_df['weight'].values)**2 / np.sum(q_df['weight'].values**2)
    N_g = np.sum(g_df['weight'].values)**2 / np.sum(g_df['weight'].values**2)
    
    return pd.Series({'accuracy': accuracy, 'auc': auc, 'accuracy_err': accuracy_err, 'auc_err': auc_err, 'q_eff': q_eff, 'g_eff': g_eff, 'N_q': N_q, 'N_g': N_g})
    
def main(args):
    def str_to_eta(x):
        x = x.split(']')[0].replace('(', '').replace(' ', '').split(',')
        return (float(x[0]) + float(x[1])) / 2
    
    def str_to_pt(x):
        x = x.split(']')[1].replace('(', '').replace(' ', '').split(',')
        return (float(x[0]) + float(x[1])) / 2
    
    save_path = os.path.dirname(args.load_path)
    save_path = os.path.join(save_path, '2D_metric_plots')
    os.makedirs(os.path.join(save_path, 'png'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'pdf'), exist_ok=True)
    
    for score in ['idepart-m', 'ipart-m', 'pfn-m', 'particle_net-m', 'efn', 'fc-reduced', 'fc_crafted']:
        df = pd.read_csv(args.load_path)
        df['score'] = df[score + '_score']
        df = df[['jets_pt', 'jets_eta', 'label', 'weight', 'score']]
        df['pred'] = np.where(df['score'] > THRESHOLD, 1, 0)
        df['jets_eta'] = df['jets_eta'].abs()
        df['jets_pt'] = df['jets_pt'] * 1e-3
        
        df['eta_bin'] = pd.cut(df['jets_eta'], bins=[0, 0.5, 1.0, 1.5, 2.1])
        df['pt_bin'] = pd.cut(df['jets_pt'], bins=[200, 300, 400, 600, 850, 1100, 1400, 1750, 2500])
        df['eta_pt_bin'] = df['eta_bin'].astype(str) + df['pt_bin'].astype(str)
        df = df.groupby(['eta_pt_bin']).apply(calc_accuracy).reset_index()
        df['eta'] = df['eta_pt_bin'].apply(str_to_eta)
        df['pt'] = df['eta_pt_bin'].apply(str_to_pt).astype(int)
        print(df.head())
        labels = ['Accuracy', 'AUC', 'Accuracy Error', 'AUC Error', 'Quark Efficiency', 'Gluon Efficiency', '$N$ quarks', '$N$ gluons']
        metrics = ['accuracy', 'auc', 'accuracy_err', 'auc_err', 'q_eff', 'g_eff', 'N_q', 'N_g']
        for label, metric in zip(labels, metrics):
            sm_df = df[['eta', 'pt', metric]]
            sm_df = sm_df.pivot(index='eta', columns='pt', values=metric)
            vmax = sm_df.max().max()
            vmin = sm_df.min().min()
            ax = sns.heatmap(sm_df, cmap='viridis', vmin=vmin, vmax=vmax, cbar_kws={'label': label})
            plt.xlabel(r'$p_{\mathrm{T}}$ [GeV]', horizontalalignment='right', x=1.0, fontsize=12)
            plt.ylabel(r'$|\eta|$', horizontalalignment='right', x=1.0, fontsize=12)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.tick_params(axis='both', which='minor', labelsize=12)

            atlasify.atlasify(
                axes=ax,
                outside=True,
                subtext=f'13 TeV, Pythia8, {MODEL_NAMING_SCHEMA[score]}\n' + r'anti-$k_{\mathrm{T}}$, $R = 0.4$ PFlow jets',
                atlas='Simulation Preliminary',
                font_size=12,
                sub_font_size=12,
                label_font_size=12,
            )
            plt.savefig(f"{save_path}/png/{metric}_{score}.png", bbox_inches='tight', dpi=400)
            plt.savefig(f"{save_path}/pdf/{metric}_{score}.pdf", bbox_inches='tight')
            plt.close()
            
def calc_rej(metric: RejectionAtFixedWorkingPoint):
    def calc_rejection(df):
        results = {}
        for score in df.columns:
            if '_score' not in score:
                continue
            metric_value = metric(y_true=df['label'].values, y_pred=df[score].values, sample_weight=df['weight'].values).numpy()
            results[score.replace('_score', '')] = metric_value
        N_g = np.sum(df[df['label'] == 0]['weight'].values)**2 / np.sum(df[df['label'] == 0]['weight'].values**2)
        results['n_count'] = N_g
        return pd.Series(results)
    return calc_rejection

def main2(args):
    save_path = os.path.dirname(args.load_path)
    save_path = os.path.join(save_path, 'perJZ_plots')
    os.makedirs(os.path.join(save_path, 'png'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'pdf'), exist_ok=True)
    
    for jz in [3, 4, 5, 6, 7]:
        df = pd.read_csv(args.load_path)
        df = df.query(f'JZ_slice == {jz}')
        df['jets_pt'] = df['jets_pt'] * 1e-6
        df['pt_bin'] = pd.cut(df['jets_pt'], bins=[0.2, 0.3, 0.4, 0.6, 0.85, 1.1, 1.4, 1.75, 2.5])
        
        metric = RejectionAtFixedWorkingPoint(name='gluon_rejection_at_quark_80wp', fixed_label_id=1, working_point=0.8, returned_label_id=0)
        df = df.groupby(['pt_bin']).apply(calc_rej(metric)).reset_index()
        print(df)
        bin_mid = df['pt_bin'].apply(lambda x: x.mid).values
        bin_width = df['pt_bin'].apply(lambda x: x.right - x.left).values
        
        plot = puma.VarVsVarPlot(
            ylabel=r'$\varepsilon_g^{-1}$',
            xlabel=r'$p_{\mathrm{T}}$ [TeV]',
            logy=False,
            logx=False,
            xmin=0.2,
            xmax=2.5,
            figsize=(11.5, 8.),
            atlas_second_tag=f'13 TeV, Pythia8, 80% WP, JZ{jz}\n' + r'anti-$k_{\mathrm{T}}$, $R = 0.4$ PFlow jets',
            atlas_first_tag='Simulation Preliminary',
            leg_loc='upper right',
            label_fontsize=30,
            fontsize=24,
            leg_fontsize=24,
            atlas_fontsize=24,
            leg_ncol=2,
        )
        
        markers = ['o', 's', 'v', 'D', 'P', 'X', 'd', 'p', 'h', '8', '>', '<', '^', '*', '+', '8']
        n_counts = 'n_count'
        colours = sns.color_palette('colorblind', 7)
        for j, name in enumerate(['idepart', 'ipart', 'particle_net', 'fc_crafted', 'fc', 'pfn', 'efn']):
            x_var = bin_mid
            x_width = bin_width
            y_var_mean = df[name].values
            counts = df[n_counts].values
            y_var_std = np.sqrt(1/y_var_mean * (1 - 1/y_var_mean) / counts) * y_var_mean**2
            is_not_nan = ~np.isnan(y_var_std)
            
            x_var = x_var[is_not_nan]
            y_var_mean = y_var_mean[is_not_nan]
            x_width = x_width[is_not_nan]
            y_var_std = y_var_std[is_not_nan]
            if jz != 7:
                y_var_mean = y_var_mean[:-1]
                x_var = x_var[:-1]
                x_width = x_width[:-1]
                y_var_std = y_var_std[:-1]
            
            plot.add(
                puma.VarVsVar(
                    x_var=x_var,
                    x_var_widths=x_width,
                    y_var_mean=y_var_mean,
                    y_var_std=y_var_std,
                    plot_y_std=True,
                    marker=markers[j],
                    markersize=12,
                    markeredgewidth=40,
                    linewidth=1.6,
                    is_marker=True,
                    label=MODEL_NAMING_SCHEMA[name],
                    colour=colours[j],
                    # linestyle='-',
                ),
            )

        plot.draw()
        os.makedirs(os.path.join(save_path, 'png'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'pdf'), exist_ok=True)
        plot.savefig(os.path.join(save_path, 'png', f'{jz}.png'), dpi=400)
        plot.savefig(os.path.join(save_path, 'pdf', f'{jz}.pdf'))
            
        
        # labels = ['Accuracy', 'AUC', 'Accuracy Error', 'AUC Error', 'Quark Efficiency', 'Gluon Efficiency', '$N$ quarks', '$N$ gluons']
        # metrics = ['accuracy', 'auc', 'accuracy_err', 'auc_err', 'q_eff', 'g_eff', 'N_q', 'N_g']
        # for label, metric in zip(labels, metrics):
        #     sm_df = df[['eta', 'pt', metric]]
        #     sm_df = sm_df.pivot(index='eta', columns='pt', values=metric)
        #     vmax = sm_df.max().max()
        #     vmin = sm_df.min().min()
        #     ax = sns.heatmap(sm_df, cmap='viridis', vmin=vmin, vmax=vmax, cbar_kws={'label': label})
        #     plt.xlabel(r'$p_{\mathrm{T}}$ [GeV]', horizontalalignment='right', x=1.0, fontsize=12)
        #     plt.ylabel(r'$|\eta|$', horizontalalignment='right', x=1.0, fontsize=12)
        #     ax.tick_params(axis='both', which='major', labelsize=12)
        #     ax.tick_params(axis='both', which='minor', labelsize=12)

        #     atlasify.atlasify(
        #         axes=ax,
        #         outside=True,
        #         subtext=f'13 TeV, Pythia8, {MODEL_NAMING_SCHEMA[score]}\n' + r'anti-$k_{\mathrm{T}}$, $R = 0.4$ PFlow jets',
        #         atlas='Simulation Preliminary',
        #         font_size=12,
        #         sub_font_size=12,
        #         label_font_size=12,
        #     )
        #     plt.savefig(f"{save_path}/png/{metric}_{score}.png", bbox_inches='tight', dpi=400)
        #     plt.savefig(f"{save_path}/pdf/{metric}_{score}.pdf", bbox_inches='tight')
        #     plt.close()

if __name__ == "__main__":
    # HUE_MAPPER = {1: 'quark', 2: 'quark', 3: 'quark', 4: 'quark', 5: 'quark', 6: 'quark', 21: 'gluon'}
    args  = parser.parse_args()
    main2(args)