import os
import sys
sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
import atlasify
import pickle
import hashlib
from functools import partial
import seaborn as sns
from hydra.core.config_store import ConfigStore
from multiprocessing import Pool
import pandas as pd
import hydra
import numpy as np
import tensorflow as tf
import keras
from typing import Optional

from jidenn.const import METRIC_NAMING_SCHEMA, LATEX_NAMING_CONVENTION, MODEL_NAMING_SCHEMA
from jidenn.evaluation.evaluator import evaluate_multiple_models, calculate_binned_metrics
from jidenn.evaluation.WorkingPoint import WorkingPoint
from jidenn.evaluation.evaluation_metrics import calculate_metrics
from jidenn.data.get_dataset import get_preprocessed_dataset
from jidenn.evaluation.plotter import plot_validation_figs, plot_data_distributions, plot_var_dependence
from jidenn.config import attn_plot_config
from jidenn.data.JIDENNDataset import JIDENNDataset, ROOTVariables
from jidenn.evaluation.evaluation_metrics import EffectiveTaggingEfficiency
from jidenn.model_builders.LearningRateSchedulers import LinearWarmup
from jidenn.const import LATEX_NAMING_CONVENTION
from jidenn.data.TrainInput import input_classes_lookup
from jidenn.data.JIDENNDataset import JIDENNDataset

CUSTOM_OBJECTS = {'LinearWarmup': LinearWarmup,
                  'EffectiveTaggingEfficiency': EffectiveTaggingEfficiency}


def plot_attn_2d(array_2d: np.ndarray,
                 mask: np.ndarray,
                 clabel: str,
                 subtext: str = '',
                 save_path: str = None,
                 max_const: int = 50,
                 vmin: Optional[float] = None,
                 vmax: Optional[float] = None,
                 ax: Optional[plt.Axes] = None,
                 fontsize: int = 20,) -> None:

    ax = sns.heatmap(array_2d, mask=mask, vmin=vmin, vmax=vmax, ax=ax)
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
    if save_path is not None:
        plt.savefig(f'{save_path}.png', dpi=400, bbox_inches='tight')
        plt.savefig(f'{save_path}.pdf', bbox_inches='tight')
        plt.close()


def attn_output_model(model_path: str, transformer_layer_index: int) -> keras.Model:
    model: keras.Model = keras.models.load_model(
        model_path, custom_objects=CUSTOM_OBJECTS, compile=False)
    model = keras.Model(
        inputs=model.inputs, outputs=model.layers[transformer_layer_index].output)
    model.summary()
    return model


@hydra.main(version_base="1.2", config_path="../jidenn/yaml_config", config_name="eval_config")
def main(args: attn_plot_config.AttnPlotConfig) -> None:
    args.data.path = '/home/jankovys/JIDENN/data/r22/forward_lead_sublead/test'
    model_path = '/home/jankovys/JIDENN/logs/r22_forward_lead_sublead/depart-no-int/model'
    INPUT_DATA_TYPE = 'constituents'
    TOTAL_SIZE = 100_000
    N_LAYERS = 11
    HEADS = 8
    CONSTS = 50
    BATCH_SIZE = 512
    save_dir = 'tmp'
    save_name = 'attn'
    os.makedirs(save_dir, exist_ok=True)
    config_hash = hashlib.sha256(f'{args}-{model_path}-{TOTAL_SIZE}-{N_LAYERS}-{HEADS}-{CONSTS}-{BATCH_SIZE}-{INPUT_DATA_TYPE}'.encode('utf-8')).hexdigest()
    # data_path = os.path.join(args.data.path, args.test_subfolder) if args.test_subfolder is not None else args.data.path
    try:
        print('Loading pickle')
        with open(f'{save_dir}/attn_{config_hash}.pkl', 'rb') as f:
            avg_attns = pickle.load(f)
    except FileNotFoundError:
        dataset = get_preprocessed_dataset(
            args_data=args.data, input_creator=None, shuffle_reading=False)
        train_input_class = input_classes_lookup(INPUT_DATA_TYPE)
        train_input_class = train_input_class()
        model_input = tf.function(func=train_input_class)
        dataset = dataset.remap_data(model_input)
        dataset = dataset.get_prepared_dataset(
            batch_size=BATCH_SIZE, take=TOTAL_SIZE)
        # model = attn_output_model(args.model_path, args.model.transformer_layer_index)
        model: keras.Model = keras.models.load_model(
            model_path, custom_objects=CUSTOM_OBJECTS, compile=False)
        print(model.summary())
        print(model.layers)
        model = keras.Model(inputs=model.inputs,
                            outputs=model.layers[-4].output)
        dataset = dataset.map(lambda x, y, z: x)

        attns = [tf.zeros((HEADS, CONSTS, CONSTS)) for _ in range(N_LAYERS)]
        sums = [tf.zeros((HEADS, CONSTS, CONSTS)) for _ in range(N_LAYERS)]
        for batch in dataset:
            output = model(batch)
            for layer in range(N_LAYERS):
                attn = output[1][layer]
                attns[layer] += tf.pad(tf.reduce_sum(attn, axis=0), [
                    [0, 0], [0, 100-attn.shape[1]], [0, 100-attn.shape[2]]])[:, :CONSTS, :CONSTS]
                non_zeros = tf.where(attns[layer] != 0, 1., 0.)
                sums[layer] += non_zeros

        avg_attns = [attns[i] / sums[i] for i in range(N_LAYERS)]
        #pickle dump
        with open(f'{save_dir}/attn_{config_hash}.pkl', 'wb') as f:
            pickle.dump(avg_attns, f)

    _, axs = plt.subplots(N_LAYERS, HEADS, figsize=(20, 30))
    # set fig taitle 
    for layer in range(N_LAYERS):
        for head in range(HEADS):
            ax = sns.heatmap(avg_attns[layer][head,  :,  :].numpy(), ax=axs[layer, head], cbar=False)
            ax.invert_yaxis()
            # remove ticks
            ax.set_xticks([])
            ax.set_yticks([])
            if layer == 0:
                ax.set_title(f'Head {head+1}')
            if head == 0:
                ax.set_ylabel(f'Layer {layer+1}')
                ax.set_yticks([0, 10, 20, 30, 40, 50])
                ax.set_yticklabels([0, 10, 20, 30, 40, 50])
            if layer == N_LAYERS-1:
                ax.set_xticks([0, 10, 20, 30, 40, 50])
                ax.set_xticklabels([0, 10, 20, 30, 40, 50])
                
    plt.savefig(f'{save_dir}/{save_name}.png', dpi=500, bbox_inches='tight')
    plt.savefig(f'{save_dir}/{save_name}.pdf', bbox_inches='tight')
    plt.close()
    
if __name__ == "__main__":
    main()
