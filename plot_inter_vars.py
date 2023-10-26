import tensorflow as tf
import tensorflow_datasets as tfds
import atlasify 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import pickle
import os

from jidenn.data.JIDENNDataset import JIDENNDataset
from jidenn.data.TrainInput import input_classes_lookup
from jidenn.const import LATEX_NAMING_CONVENTION

parser = argparse.ArgumentParser()
parser.add_argument('--load_path', type=str)
parser.add_argument('--save_path', type=str)

FONTSIZE = 20

def pad(max_const):
    @tf.function
    def _pad(data, weight):
        weight = tf.cast(weight, tf.float32)
        weight = tf.fill(tf.shape(data[1]['z']), weight)
        weight = tf.pad(weight, [[0,max_const-tf.shape(weight)[1]],[0,max_const-tf.shape(weight)[1]]])
        data = {var: tf.pad(val, [[0,max_const-tf.shape(val)[1]],[0,max_const-tf.shape(val)[1]]]) for var, val in data[1].items()}
        return data, weight
    return _pad

# def pad(max_const):
#     @tf.function
#     def _pad(data, weight):
#         weight = tf.cast(weight, tf.float32)
#         weight = tf.fill(tf.shape(data[1]['z']), weight)
#         weight = tf.pad(weight, [[0,max_const-tf.shape(weight)[1]],[0,max_const-tf.shape(weight)[1]]])
#         data = {var: tf.pad(val, [[0,max_const-tf.shape(val)[1]],[0,max_const-tf.shape(val)[1]]]) for var, val in data[1].items()}
#         for var in data.keys():
#             if var == 'z':
#                 continue
#             data[var] = tf.exp(data[var])
#         return data, weight
#     return _pad

# def pad(max_const):
#     @tf.function
#     def _pad(data, weight):
#         weight = tf.cast(weight, tf.float32)
#         data = {var: tf.pad(val, [[0,max_const-tf.shape(val)[1]],[0,max_const-tf.shape(val)[1]]]) for var, val in data[1].items()}
#         weight = tf.fill(tf.shape(data['z']), weight)
#         return data, weight
#     return _pad

@tf.function
def reduce_sum(previous, data_w):
    data, weight = data_w
    new_mean = {var: val*weight + previous[0][var] for var, val in data.items()}
    new_weight = previous[1] + weight
    return new_mean, new_weight


def plot_inter_vars(base_dataset, label, max_const, var_names=['k_t', 'm2', 'z', 'delta']):
    
    try:
        with open(os.path.join(args.save_path, f'{label}_mean.pkl'), 'rb') as f:
            mean_values_all = pickle.load(f)
        with open(os.path.join(args.save_path, f'{label}_sum_of_weights.pkl'), 'rb') as f:
            sum_of_weights = pickle.load(f)
        vmins = {var: (mean_values_all[var]/sum_of_weights).numpy().flatten()[(mean_values_all[var]/sum_of_weights).numpy().flatten().nonzero()].min() for var in var_names}
        vmaxs = {var: (mean_values_all[var]/sum_of_weights).numpy().flatten()[(mean_values_all[var]/sum_of_weights).numpy().flatten().nonzero()].max() for var in var_names}
        
        label2 = 'gluon' if label == 'quark' else 'quark'
        with open(os.path.join(args.save_path, f'{label2}_mean.pkl'), 'rb') as f:
            mean_values_all_2 = pickle.load(f)
        with open(os.path.join(args.save_path, f'{label2}_sum_of_weights.pkl'), 'rb') as f:
            sum_of_weights_2 = pickle.load(f)
        vmins2 = {var: (mean_values_all_2[var]/sum_of_weights_2).numpy().flatten()[(mean_values_all_2[var]/sum_of_weights_2).numpy().flatten().nonzero()].min() for var in var_names}
        vmaxs2 = {var: (mean_values_all_2[var]/sum_of_weights_2).numpy().flatten()[(mean_values_all_2[var]/sum_of_weights_2).numpy().flatten().nonzero()].max() for var in var_names}
        
        vmins = {var: min(vmins[var], vmins2[var]) for var in var_names}
        vmaxs = {var: max(vmaxs[var], vmaxs2[var]) for var in var_names}
            
    except FileNotFoundError:
        dataset = base_dataset.map(pad(max_const))
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        initial = {var: tf.zeros((max_const, max_const), dtype=tf.float32) for var in var_names}
        mean_values_all, sum_of_weights = dataset.reduce((initial, np.zeros((max_const,max_const), dtype=np.float32)), reduce_sum)
        os.makedirs(os.path.join(args.save_path), exist_ok=True)
        with open(os.path.join(args.save_path, f'{label}_mean.pkl'), 'wb') as f:
            pickle.dump(mean_values_all, f)
        with open(os.path.join(args.save_path, f'{label}_sum_of_weights.pkl'), 'wb') as f:
            pickle.dump(sum_of_weights, f)
        
        vmins = {var: (mean_values_all[var]/sum_of_weights).numpy().flatten()[(mean_values_all[var]/sum_of_weights).numpy().flatten().nonzero()].min() for var in var_names}
        vmaxs = {var: (mean_values_all[var]/sum_of_weights).numpy().flatten()[(mean_values_all[var]/sum_of_weights).numpy().flatten().nonzero()].max() for var in var_names}
    
    os.makedirs(os.path.join(args.save_path, 'png'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'pdf'), exist_ok=True)
    
    for var in var_names:
        print('Plotting', var, 'for', label)
        mean_values = mean_values_all[var] / sum_of_weights
        mean_values = mean_values.numpy()
        ax = sns.heatmap(mean_values, mask=np.diag(np.ones(max_const)), vmin=vmins[var], vmax=vmaxs[var])
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.set_yticks([0, 10, 20, 30, 40, 50])
        ax.set_xticklabels([0, 10, 20, 30, 40, 50])
        ax.set_yticklabels([0,10,20,30,40,50])
        ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
        ax.tick_params(axis='both', which='minor', labelsize=FONTSIZE)
        ax.invert_yaxis()
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=FONTSIZE)
        cbar.set_label(r'$\langle ' + LATEX_NAMING_CONVENTION[var].replace('$', '') + r' \rangle $', fontsize=FONTSIZE)
        plt.xlabel('Constituent Index', horizontalalignment='right', x=1.0, fontsize=FONTSIZE)
        plt.ylabel('Constituent Index', horizontalalignment='right', y=1.0, fontsize=FONTSIZE)

        atlasify.atlasify(atlas="Simulation Internal", 
                        outside=True, 
                        subtext=f"13 TeV, Pythia8, {label}\n" + r"anti-$k_{\mathrm{T}}$, $R = 0.4$ PFlow jets",
                        font_size=FONTSIZE,
                        sub_font_size=FONTSIZE,
                        label_font_size=FONTSIZE,
                        )
        plt.savefig(os.path.join(args.save_path, 'png', f'{var}_{label}_mean.png'), dpi=400, bbox_inches='tight')
        plt.savefig(os.path.join(args.save_path, 'pdf', f'{var}_{label}_mean.pdf'), bbox_inches='tight')
        plt.close()
        
        if label == 'quark':
            try:
                with open(os.path.join(args.save_path, f'gluon_mean.pkl'), 'rb') as f:
                    g_mean_values_all = pickle.load(f)
                with open(os.path.join(args.save_path, f'gluon_sum_of_weights.pkl'), 'rb') as f:
                    g_sum_of_weights = pickle.load(f)
                
                g_mean_values = g_mean_values_all[var] / g_sum_of_weights
                g_mean_values = g_mean_values.numpy()
                diff = g_mean_values - mean_values
                ratio = g_mean_values / mean_values
                
                for name, values, sign in zip(['diff', 'ratio'], [diff, ratio], ['-', '/']):
                    ax = sns.heatmap(values, mask=np.diag(np.ones(max_const)))
                    ax.set_xticks([0, 10, 20, 30, 40, 50])
                    ax.set_yticks([0, 10, 20, 30, 40, 50])
                    ax.set_xticklabels([0, 10, 20, 30, 40, 50])
                    ax.set_yticklabels([0,10,20,30,40,50])
                    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
                    ax.tick_params(axis='both', which='minor', labelsize=FONTSIZE)
                    ax.invert_yaxis()
                    cbar = ax.collections[0].colorbar
                    cbar.ax.tick_params(labelsize=FONTSIZE)
                    cbar.set_label('$' + r'\langle ' + LATEX_NAMING_CONVENTION[var].replace('$', '') + r' \rangle_{\mathrm{gluon}}' + f' {sign} ' + r'\langle ' + LATEX_NAMING_CONVENTION[var].replace('$', '') + r' \rangle_{\mathrm{quark}}' +'$', fontsize=FONTSIZE)
                    plt.xlabel('Constituent Index', horizontalalignment='right', x=1.0, fontsize=FONTSIZE)
                    plt.ylabel('Constituent Index', horizontalalignment='right', y=1.0, fontsize=FONTSIZE)

                    atlasify.atlasify(atlas="Simulation Internal", 
                                    outside=True, 
                                    subtext=f"13 TeV, Pythia8\n" + r"anti-$k_{\mathrm{T}}$, $R = 0.4$ PFlow jets",
                                    font_size=FONTSIZE,
                                    sub_font_size=FONTSIZE,
                                    label_font_size=FONTSIZE,
                                    )
                    plt.savefig(os.path.join(args.save_path, 'png', f'{var}_{name}_mean.png'), dpi=400, bbox_inches='tight')
                    plt.savefig(os.path.join(args.save_path, 'pdf', f'{var}_{name}_mean.pdf'), bbox_inches='tight')
                    plt.close()
                
                
            except FileNotFoundError:
                continue
                    
                
            


def main(args):
    MAX_CONSTITUENTS = 50+1
    TAKE = 10_000_000

    full_dataset = JIDENNDataset.load(args.load_path)
    full_dataset = full_dataset.take(TAKE)
    inputs = input_classes_lookup('i_c')()
    inputs.max_constituents = MAX_CONSTITUENTS
    
    for label in ['gluon', 'quark']:
        if label == 'gluon':
            dataset = full_dataset.filter(lambda x: tf.reduce_any(tf.equal(x['jets_PartonTruthLabelID'], [21])))
        elif label == 'quark':
            dataset = full_dataset.filter(lambda x: tf.reduce_any(tf.equal(x['jets_PartonTruthLabelID'], [1,2,3,4,5])))
        else:
            raise ValueError('Unknown label')
            
        dataset = dataset.dataset.map(lambda x: (x, x['weight_spectrum']))
        dataset = dataset.map(lambda x, y: (inputs(x), y))
        plot_inter_vars(dataset, label, MAX_CONSTITUENTS)
            
    
    
    
    
if __name__ == '__main__':
    args = parser.parse_args()
    if args.load_path is None:
        args.load_path = '/home/jankovys/JIDENN/data/pythia_nW_flat_100_JZ7_cut/train'
    if args.save_path is None:
        args.save_path = '/home/jankovys/JIDENN/data/pythia_nW_flat_100_JZ7_cut/train/inter_2Dplots'
        

    main(args)
    print('Done!')