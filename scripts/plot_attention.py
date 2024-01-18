import os
import sys
sys.path.append(os.getcwd())
from typing import Optional
import tensorflow as tf
import numpy as np
import hydra
import pandas as pd
from multiprocessing import Pool
from hydra.core.config_store import ConfigStore
import seaborn as sns
from functools import partial
import hashlib
import pickle
import atlasify
import matplotlib.pyplot as plt

from jidenn.data.JIDENNDataset import JIDENNDataset
from jidenn.data.TrainInput import input_classes_lookup
from jidenn.const import LATEX_NAMING_CONVENTION
from jidenn.model_builders.LearningRateSchedulers import LinearWarmup
from jidenn.evaluation.evaluation_metrics import EffectiveTaggingEfficiency

from jidenn.data.JIDENNDataset import JIDENNDataset, ROOTVariables
from jidenn.config import attn_plot_config
from jidenn.evaluation.plotter import plot_validation_figs, plot_data_distributions, plot_var_dependence
from jidenn.data.get_dataset import get_preprocessed_dataset
from jidenn.model_builders.LearningRateSchedulers import LinearWarmup
from jidenn.evaluation.evaluation_metrics import EffectiveTaggingEfficiency
from jidenn.evaluation.evaluation_metrics import calculate_metrics
from jidenn.evaluation.WorkingPoint import WorkingPoint
from jidenn.evaluation.evaluator import evaluate_multiple_models, calculate_binned_metrics
from jidenn.const import METRIC_NAMING_SCHEMA, LATEX_NAMING_CONVENTION, MODEL_NAMING_SCHEMA

CUSTOM_OBJECTS = {'LinearWarmup': LinearWarmup,
             'EffectiveTaggingEfficiency': EffectiveTaggingEfficiency}


def plot_attn_2d(array_2d: np.ndarray, 
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
    plt.xlabel('Constituent Index', horizontalalignment='right', x=1.0, fontsize=fontsize)
    plt.ylabel('Constituent Index', horizontalalignment='right', y=1.0, fontsize=fontsize)

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

def attn_output_model(model_path: str, transformer_layer_index: int) -> tf.keras.Model:
    model: tf.keras.Model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS, compile=False)
    model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[transformer_layer_index].output)
    model.summary()
    return model


@hydra.main(version_base="1.2", config_path="jidenn/yaml_config", config_name="eval_config")
def main(args: attn_plot_config.AttnPlotConfig) -> None:
    
    data_path = os.path.join(args.data.path, args.test_subfolder) if args.test_subfolder is not None else args.data.path
    dataset = get_preprocessed_dataset(file=data_path, args_data=args.data,
                                           input_creator=None, shuffle_reading=False)