import tensorflow as tf
import uproot
import awkward as ak
import numpy as np
import os
import h5py


def get_jet_h5_iterator(filename, dataset, variable_names):
    def _h5_iterator():
        with h5py.File(filename, 'r') as hf:
            for dato in hf[dataset]:
                yield {name: var for name, var in zip(variable_names, dato)}
    return _h5_iterator


def get_flow_h5_iterator(filename, dataset, variable_names):
    def _h5_iterator():
        with h5py.File(filename, 'r') as hf:
            for dato in hf[dataset]:
                jet = list(zip(*dato))
                is_good = np.array(jet[-1], dtype=np.int32)
                yield {name: np.array(var)[np.where(is_good)] for name, var in zip(variable_names, jet)}
    return _h5_iterator


def convert_h5_to_tfdataset(load_path,
                            jet_dataset='jets',
                            jet_variables=['jets_E', 'jets_eta', 'jets_pt', 'jets_phi',
                                           'jets_label', 'jets_num', 'event', 'mu', 'corr_mu'],
                            jet_types=[tf.float32, tf.float32, tf.float32, tf.float32,
                                       tf.int32, tf.float32, tf.int32, tf.float32, tf.float32],
                            flow_dataset='flow',
                            flow_variables=["jets_UFO_pt", "jets_UFO_energy", "jets_UFO_deta", "jets_UFO_dphi",
                                            "jets_UFO_dr", "jets_UFO_track_pt", "jets_UFO_d0", "jets_UFO_z0SinTheta"],
                            flow_types=[tf.float32, tf.float32, tf.float32, tf.float32,
                                        tf.float32, tf.float32, tf.float32, tf.float32],
                            ):

    jet_specs = {name: tf.TensorSpec(shape=(), dtype=tp) for name, tp in zip(jet_variables, jet_types)}
    jet_dataset = tf.data.Dataset.from_generator(
        get_jet_h5_iterator(load_path, jet_dataset, variable_names=jet_variables),
        output_signature=jet_specs
    )
    if flow_dataset is None:
        return jet_dataset
    flow_specs = {name: tf.TensorSpec(shape=(None, ), dtype=tp) for name, tp in zip(flow_variables, flow_types)}

    flow_dataset = tf.data.Dataset.from_generator(
        get_flow_h5_iterator(load_path, flow_dataset, variable_names=flow_variables),
        output_signature=flow_specs
    )

    return tf.data.Dataset.zip((jet_dataset, flow_dataset)).map(lambda x, y: {**x, **y})
