import os
import sys
sys.path.append(os.getcwd())
import tensorflow as tf
from dataclasses import dataclass
import numpy as np
import pandas as pd
from multiprocessing import Pool
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
#
from jidenn.data.JIDENNDataset import JIDENNDataset, ROOTVariables
from jidenn.data.TrainInput import input_classes_lookup
from jidenn.data.get_dataset import get_preprocessed_dataset
from jidenn.model_builders.LearningRateSchedulers import LinearWarmup
from jidenn.evaluation.evaluation_metrics import EffectiveTaggingEfficiency


@dataclass
class Data:
    path = None
    target = 'jets_PartonTruthLabelID'
    target_labels = [[21], [1, 2, 3, 4, 5, 6]]
    labels = ['gluon', 'quark']
    variable_unknown_labels = [-1, -999]
    label_weights = None
    weight = None
    cut = '(jets_eta<2.1) && (jets_eta>-2.1)'


CUSTOM_OBJECTS = {'LinearWarmup': LinearWarmup,
                  'EffectiveTaggingEfficiency': EffectiveTaggingEfficiency}

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="Path to the model.")
parser.add_argument("--dataset_path", type=str, help="Path to the dataset.")
parser.add_argument("--input_type", type=str, default="highlevel", help="names of the models.")
parser.add_argument("--save_name", type=str, default="feature_importance", help="Name of the save files.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
parser.add_argument("--take", type=int, default=2_000_000, help="Number of jets to take.")
parser.add_argument("--weight", type=str, default=None, help="Weight variable.")

VARIABLES = [
    'jets_ActiveArea4vec_eta',
    'jets_ActiveArea4vec_m',
    'jets_ActiveArea4vec_phi',
    'jets_ActiveArea4vec_pt',
    'jets_DetectorEta',
    'jets_FracSamplingMax',
    'jets_FracSamplingMaxIndex',
    'jets_GhostMuonSegmentCount',
    'jets_JVFCorr',
    'jets_JetConstitScaleMomentum_eta',
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
    'corrected_averageInteractionsPerCrossing',
    'jets_PFO_n',
    'jets_ChargedPFOWidthPt1000',
    'jets_TrackWidthPt1000',
    'jets_NumChargedPFOPt1000',
    'jets_SumPtChargedPFOPt500',
    'jets_NumChargedPFOPt500',]


def get_noisify_fn(variable, mean, std):
    @tf.function
    def noisify(sample: ROOTVariables) -> ROOTVariables:
        sample = sample.copy()
        sample[variable] = tf.random.uniform(tf.shape(sample[variable]), minval=mean - tf.sqrt(3.) * std, maxval=mean + tf.sqrt(3.) * std)
        return sample
    return noisify


def main(args: argparse.Namespace) -> None:
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)

    data_config = Data()
    data_config.path = args.dataset_path
    data_config.weight = args.weight

    dataset = get_preprocessed_dataset(file=args.dataset_path, args_data=data_config,
                                       input_creator=None, shuffle_reading=False)

    train_input_class = input_classes_lookup(args.input_type)
    train_input_class = train_input_class()
    model_input = tf.function(func=train_input_class)
    dataset = dataset.remap_data(model_input)
    model: tf.keras.Model = tf.keras.models.load_model(args.model_path, custom_objects=CUSTOM_OBJECTS)
    means = model.layers[1].mean[0, :].numpy()
    stds = model.layers[1].variance[0, :].numpy()
    stds = np.sqrt(stds)

    ds = dataset.get_prepared_dataset(batch_size=args.batch_size,
                                      ragged=False,
                                      take=args.take)
    score = model.evaluate(ds, return_dict=True, use_multiprocessing=True)
    nominal_accuracy = score['binary_accuracy']
    print(f"Accuracy: {nominal_accuracy}")

    print("Noisifying")
    df = {}
    for i, variable in enumerate(VARIABLES):
        ds = dataset.remap_data(get_noisify_fn(variable=variable, mean=means[i], std=stds[i]))
        ds = ds.get_prepared_dataset(batch_size=args.batch_size,
                                     ragged=False,
                                     take=args.take)
        score = model.evaluate(ds, return_dict=True, use_multiprocessing=True)
        accuracy = score['binary_accuracy']
        accuracy_difference = nominal_accuracy - accuracy
        df[variable] = accuracy_difference

    df = pd.DataFrame.from_dict(df, orient='index', columns=['accuracy_difference'])
    df = df.sort_values(by='accuracy_difference', ascending=False)
    df.to_csv(os.path.join(args.model_path.replace('/model', ''), f'{args.save_name}.csv'))
    print(df)
    # plot feature importance as a bar plot
    plt.figure(figsize=(10,8))
    ax = sns.barplot(x="accuracy_difference", y=df.index, data=df)
    ax.set(xlabel="Accuracy difference", ylabel=None)
    plt.savefig(os.path.join(args.model_path.replace('/model', ''),
                f'{args.save_name}.png'), dpi=400, bbox_inches='tight')

    ##########################################################################################################################################


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
