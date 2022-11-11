import tensorflow as tf
import uproot
import pandas as pd
import vector
import os
import pickle
import argparse
import awkward as ak
from typing import Callable

parser = argparse.ArgumentParser()
parser.add_argument("--root_file", type=str, help="File to transofrm to tf dataset")
parser.add_argument("--save_folder", type=str, default=None, help="Folder to save the tf dataset")

ROOT_variables = dict[str, tf.RaggedTensor]

def get_PFO_JetFrame(sample: ROOT_variables) -> ROOT_variables:
    vector.register_awkward()
    Lvector = ak.Array({
        "pt": sample['jets_pt'].to_list(),
        "eta": sample['jets_eta'].to_list(),
        "phi": sample['jets_phi'].to_list(),
        "m": sample['jets_m'].to_list(),
    }, with_name="Momentum4D")

    awk = ak.Array({'pt': sample['jets_PFO_pt'].to_list(),
                    'eta': sample['jets_PFO_eta'].to_list(),
                    'phi': sample['jets_PFO_phi'].to_list(),
                    'm': sample['jets_PFO_m'].to_list(),
                    }, with_name="Momentum4D")
    boosted_awk = awk.boost(-Lvector)
    sample.update({f'jets_PFO_{var}_JetFrame': tf.ragged.constant(ak.to_list(getattr(boosted_awk, var)))
                  for var in ['pt', 'eta', 'phi', 'm']})
    return sample


def no_pile_up(sample: ROOT_variables) -> ROOT_variables:
    jet_mask = tf.math.greater(sample['jets_PartonTruthLabelID'], 0)
    event_mask = tf.reduce_any(jet_mask, axis=-1)
    for key, item in sample.items():
        if key.startswith('jets_') and key != 'jets_n':
            sample[key] = tf.ragged.boolean_mask(item, jet_mask)
        sample[key] = tf.ragged.boolean_mask(sample[key], event_mask)
    return sample

# function that converts a root file to a tf dataset


def root_file_to_dataset(filename: str, transformation: Callable[[ROOT_variables], ROOT_variables] | None = None) -> tf.data.Dataset:
    file = uproot.open(filename)
    tree = file['NOMINAL']
    metadata = file['h_metadata'].values()
    variables = tree.keys()
    sample = {}
    for var in variables:
        df = tree[var].array(library="pd")
        if df.empty:
            continue
        if df.index.nlevels > 1:
            df = df.groupby(level=0).apply(list)
        sample[var] = tf.ragged.constant(df)

    sample = transformation(sample) if transformation is not None else sample
    sample['metadata'] = tf.tile(tf.constant(metadata)[tf.newaxis, :], [sample['eventNumber'].shape[0], 1])
    dataset = tf.data.Dataset.from_tensor_slices(sample)
    return dataset


def save_dataset(dataset: tf.data.Dataset, save_path: str):
    element_spec = dataset.element_spec
    tf.data.experimental.save(dataset, save_path, compression='GZIP')
    with open(os.path.join(save_path, 'element_spec'), 'wb') as f:
        pickle.dump(element_spec, f)


def load_dataset(file_path: str) -> tf.data.Dataset:
    with open(os.path.join(file_path, 'element_spec'), 'rb') as f:
        element_spec = pickle.load(f)
    dataset = tf.data.experimental.load(file_path, compression='GZIP', element_spec=element_spec)
    return dataset

def main(args: argparse.Namespace):
    filename = args.root_file
    base_path = os.path.dirname(filename)
    if args.save_folder is None:
        save_file_name = filename.split("/")[-1].split(".")[-2]
        save_path = os.path.join(base_path, save_file_name)
    else:
        save_path = args.save_folder
    os.makedirs(save_path, exist_ok=True)
    dataset = root_file_to_dataset(filename, transformation=lambda x: get_PFO_JetFrame(no_pile_up(x)))
    save_dataset(dataset, save_path)
    dataset = load_dataset(save_path)
    for i in dataset.take(1):
        print(i)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

