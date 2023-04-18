import tensorflow as tf
import tensorflow_addons as tfa  # this is necessary for the LAMB optimizer to work
import numpy as np
import os
import logging
import pandas as pd
import time
import matplotlib.ticker as mticker
import plotly.graph_objects as go
#
import jidenn.data.data_info as data_info
from jidenn.config import eval_config
from jidenn.evaluation.plotter import plot_validation_figs, plot_metrics_per_cut
from jidenn.data.utils.Cut import Cut
from jidenn.data.get_dataset import get_preprocessed_dataset
from jidenn.model_builders.LearningRateSchedulers import LinearWarmup
from jidenn.data.TrainInput import input_classes_lookup
from jidenn.evaluation.evaluation_metrics import calculate_metrics
from jidenn.evaluation.variable_latex_names import LATEX_NAMING_CONVENTION

from jidenn.data.data_info import *
from jidenn.model_builders.data_augment import InfraredAugmentation


from dataclasses import dataclass


@dataclass
class Variables:
    perJet = ['jets_ActiveArea4vec_m',
              'jets_ActiveArea4vec_phi',
              'jets_ActiveArea4vec_pt',
              'jets_FracSamplingMax',
              'jets_FracSamplingMaxIndex',
              'jets_GhostMuonSegmentCount',
              'jets_JVFCorr',
              'jets_JetConstitScaleMomentum_m',
              'jets_JetConstitScaleMomentum_phi',
              'jets_JetConstitScaleMomentum_pt',
              'jets_JvtRpt',
              'jets_fJVT',
              'jets_passFJVT',
              'jets_passJVT',
              'jets_Timing',
              'jets_Jvt',
              'jets_EMFrac',
              'jets_Width',
              'jets_chf',
              'jets_eta',
              'jets_m',
              'jets_phi',
              'jets_pt',
              'jets_PFO_n',
              'jets_ChargedPFOWidthPt1000[0]',
              'jets_TrackWidthPt1000[0]',
              'jets_NumChargedPFOPt1000[0]',
              'jets_NumChargedPFOPt500[0]',
              'jets_SumPtChargedPFOPt500[0]']
    perJetTuple = ['jets_PFO_m',
                   'jets_PFO_pt',
                   'jets_PFO_eta',
                   'jets_PFO_phi',]
    perEvent = ['corrected_averageInteractionsPerCrossing[0]']


@dataclass
class Data:
    labels = ['gluon', 'quark']
    num_labels = 2
    input_size = 8
    target = 'jets_PartonTruthLabelID'
    variables = Variables()
    weight = None
    cut = "(jets_pt>20_000) && (jets_passFJVT==1) && (jets_passJVT==1) && (jets_Jvt>0.8) && (jets_PFO_n>5)"
    tttree_name = 'NOMINAL'
    gluon = 0
    quark = 1
    raw_gluon = 21
    raw_quarks = [1, 2, 3, 4, 5, 6]
    raw_unknown = [-1, -999]
    path = './data/dataset2_3'
    JZ_slices = ['JZ05_r10724']
    JZ_cut = ["jets_pt>800_000"]
    JZ_weights = [1.0]
    cached = None


np.random.seed(44)
tf.random.set_seed(44)


# train_input_class = input_classes_lookup('constituents')
# train_input_class = train_input_class()
# model_input = tf.function(func=train_input_class)


@tf.function
def model_input(sample):
    m_const = sample['perJetTuple']['jets_PFO_m']
    pt_const = sample['perJetTuple']['jets_PFO_pt']
    eta_const = sample['perJetTuple']['jets_PFO_eta']
    phi_const = sample['perJetTuple']['jets_PFO_phi']

    m_jet = sample['perJet']['jets_m']
    pt_jet = sample['perJet']['jets_pt']
    eta_jet = sample['perJet']['jets_eta']
    phi_jet = sample['perJet']['jets_phi']

    PFO_E = tf.math.sqrt(pt_const**2 + m_const**2)
    jet_E = tf.math.sqrt(pt_jet**2 + m_jet**2)
    deltaEta = eta_const - tf.math.reduce_mean(eta_jet)
    deltaPhi = phi_const - tf.math.reduce_mean(phi_jet)
    deltaR = tf.math.sqrt(deltaEta**2 + deltaPhi**2)

    logPT = tf.math.log(pt_const)

    logPT_PTjet = tf.math.log(pt_const / tf.math.reduce_mean(pt_jet))
    logE_Ejet = tf.math.log(PFO_E / tf.math.reduce_mean(jet_E))
    logE = tf.math.log(PFO_E)
    m = m_const
    # data = [logPT, logPT_PTjet, logE, logE_Ejet, m, deltaEta, deltaPhi, deltaR]
    # return {'log_PT|PTjet': logPT_PTjet, 'log_E|Ejet': logE_Ejet,
    #         'm': m, 'deltaEta': deltaEta, 'deltaPhi': deltaPhi, 'deltaR': deltaR}
    return {'eta': eta_const, 'phi': phi_const, 'pT': pt_const}

file = ['/Users/samueljankovych/Documents/MFF/bakalarka/JIDENN/data/dataset2_3/JZ05_r10724/test']
test_ds = get_preprocessed_dataset(file, Data())
test_ds = test_ds.map_data(model_input)
# augemtn = InfraredAugmentation(1, 1)
# test_ds = test_ds.map_data(tf.function(func=augemtn))
df = test_ds.apply(lambda x: x.take(5)).to_pandas()

# new_df = df[['deltaEta', 'deltaPhi', 'log_PT|PTjet', 'mask']]
new_df = df[['eta', 'phi', 'pT']]
new_df['jet_id'] = list(range(len(new_df)))
# new_df = explode_nested_variables(new_df, ['deltaEta', 'deltaPhi', 'log_PT|PTjet'])
new_df = explode_nested_variables(new_df, ['eta', 'phi', 'pT'])
print(new_df)
# sns.scatterplot(data=new_df, x='log_PT|PTjet', y='mask')
# sns.scatterplot(data=new_df, x='log_PT|PTjet', y='mask')
# plt.ylim(0., 1.)
# plt.xlim(-17., 0.)
# plt.savefig('tmp/mask.png')
# plt.close()
# sns.histplot(data=new_df, x='log_PT|PTjet', label=f'{len(new_df)}')
# plt.legend()
# plt.savefig('tmp/dist.png')
# plt.close()

# 1364 8249 18576
# ax = sns.scatterplot(data=new_df, x='deltaPhi', y='deltaEta', hue='log_PT|PTjet',
#                      size='log_PT|PTjet', sizes=(50, 400), alpha=0.8)
# sns.displot(new_df, x="phi", y="eta", weights='pT', hue='jet_id', legend=False)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
phis = new_df['phi'].to_numpy(dtype=np.float32)
etas = new_df['eta'].to_numpy(dtype=np.float32)
pts = new_df['pT'].to_numpy(dtype=np.float32)
hist, xedges, yedges = np.histogram2d(etas, phis, weights=pts, bins=200, range=[[-2.1, 2.1], [-3.14, 3.14]])



fig = go.Figure(data=[go.Surface(z=hist, x=xedges, y=yedges)])
fig.update_layout(title='jet constituents')
# fig.update_layout(scene=dict(zaxis=dict(type='log')))
fig.show()

# My axis should display 10⁻¹ but you can switch to e-notation 1.00e+01


# def log_tick_formatter(val, pos=None):
#     return f"$10^{{{int(val)}}}$"  # remove int() if you don't use MaxNLocator
#     # return f"{10**val:.2e}"      # e-Notation


# ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
# ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))
# plt.ylim(-0.5, 0.5)
# plt.xlim(-1, 1)
# plt.savefig('tmp/constituents.png')
# plt.close()
