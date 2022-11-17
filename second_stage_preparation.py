import tensorflow as tf
import os
import awkward as ak
import vector
import argparse
import logging
#

from src.data.ROOTDataset import ROOTDataset, ROOTVariables

logging.basicConfig(format='[%(asctime)s][%(levelname)s] - %(message)s',
                    level=logging.INFO,  datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--save_path", type=str, help="Path to save the dataset")
parser.add_argument("--file_path", type=str, help="Path to the root file")

tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)


def get_PFO_in_JetFrame(jets_pt: tf.Tensor,
                        jets_eta: tf.Tensor,
                        jets_phi: tf.Tensor,
                        jets_m: tf.Tensor,
                        jets_PFO_pt: tf.RaggedTensor,
                        jets_PFO_eta: tf.RaggedTensor,
                        jets_PFO_phi: tf.RaggedTensor,
                        jets_PFO_m: tf.RaggedTensor):
    vector.register_awkward()
    Lvector = ak.Array({
        "pt": jets_pt.numpy(),
        "eta": jets_eta.numpy(),
        "phi": jets_phi.numpy(),
        "m": jets_m.numpy(),
    }, with_name="Momentum4D")

    awk = ak.Array({'pt': jets_PFO_pt.to_list(),
                    'eta': jets_PFO_eta.to_list(),
                    'phi': jets_PFO_phi.to_list(),
                    'm': jets_PFO_m.to_list(),
                    }, with_name="Momentum4D")
    boosted_awk = awk.boost(-Lvector)

    return [tf.RaggedTensor.from_nested_row_lengths(ak.to_list(ak.flatten(getattr(boosted_awk, var), axis=None)), nested_row_lengths=jets_PFO_pt.nested_row_lengths(), validate=True)
            for var in ['pt', 'eta', 'phi', 'm']]


@tf.function
def py_func_get_PFO_in_JetFrame(sample: ROOTVariables) -> ROOTVariables:
    sample = sample.copy()
    input = [sample['jets_pt'], sample['jets_eta'], sample['jets_phi'], sample['jets_m'],
             sample['jets_PFO_pt'], sample['jets_PFO_eta'], sample['jets_PFO_phi'], sample['jets_PFO_m']]
    output = tf.py_function(get_PFO_in_JetFrame, inp=input, Tout=[
                            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.float32)]*4)
    sample['jets_PFO_pt_JetFrame'], sample['jets_PFO_eta_JetFrame'], sample['jets_PFO_phi_JetFrame'], sample['jets_PFO_m_JetFrame'] = output
    return sample


@tf.function
def no_pile_up(sample: ROOTVariables) -> ROOTVariables:
    sample = sample.copy()
    jet_mask = tf.math.greater(sample['jets_PartonTruthLabelID'], 0)
    for key, item in sample.items():
        if key.startswith('jets_') and key != 'jets_n':
            sample[key] = tf.ragged.boolean_mask(item, jet_mask)
    return sample

@tf.function
def filter_empty_jets(sample: ROOTVariables) -> tf.Tensor:
    return tf.greater(tf.size(sample['jets_PartonTruthLabelID']), 0)

@tf.function
def shard_func(sample:ROOTVariables) -> tf.Tensor:
    return tf.cast(sample['runNumber'] % 12, tf.int64)

def load_dataset_file(file: str, save_path: str) -> None:
    root = ROOTDataset.load(file)
    root_dt = root.dataset
    root_dt = root_dt.map(no_pile_up, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).filter(filter_empty_jets)
    # root_dt = root_dt.map(py_func_get_PFO_in_JetFrame, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    root_dt = root_dt.prefetch(tf.data.AUTOTUNE)
    variables = root.variables
    # tf.data.experimental.save(root_dt, path=save_path, compression='GZIP', shard_func=shard_func)
    ROOTDataset(root_dt, variables).save(save_path=save_path, shard_func=shard_func)

def main(args: argparse.Namespace) -> None:
    os.makedirs(args.save_path, exist_ok=True)
    load_dataset_file(args.file_path, args.save_path)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
