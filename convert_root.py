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


def root_to_preJIDENN_dataset(file: str, save_path: str) -> None:
    # def get_PFO_in_JetFrame(sample: ROOTVariables) -> ROOTVariables:
    #     vector.register_awkward()
    #     Lvector = ak.Array({
    #         "pt": sample['jets_pt'].to_list(),
    #         "eta": sample['jets_eta'].to_list(),
    #         "phi": sample['jets_phi'].to_list(),
    #         "m": sample['jets_m'].to_list(),
    #     }, with_name="Momentum4D")

    #     awk = ak.Array({'pt': sample['jets_PFO_pt'].to_list(),
    #                     'eta': sample['jets_PFO_eta'].to_list(),
    #                     'phi': sample['jets_PFO_phi'].to_list(),
    #                     'm': sample['jets_PFO_m'].to_list(),
    #                     }, with_name="Momentum4D")
    #     boosted_awk = awk.boost(-Lvector)
    #     sample.update({f'jets_PFO_{var}_JetFrame': tf.RaggedTensor.from_nested_row_lengths(ak.to_list(ak.flatten(getattr(boosted_awk, var), axis=None)), nested_row_lengths=sample[f'jets_PFO_{var}'].nested_row_lengths(), validate=False)
    #                    for var in ['pt', 'eta', 'phi', 'm']})
    #     return sample

    # def no_pile_up(sample: ROOTVariables) -> ROOTVariables:
    #     jet_mask = tf.math.greater(sample['jets_PartonTruthLabelID'], 0)
    #     event_mask = tf.reduce_any(jet_mask, axis=-1)
    #     for key, item in sample.items():
    #         if key.startswith('jets_') and key != 'jets_n':
    #             sample[key] = tf.ragged.boolean_mask(item, jet_mask)
    #         sample[key] = tf.ragged.boolean_mask(sample[key], event_mask)
    #     return sample

    # def transformation(sample: ROOTVariables) -> ROOTVariables:
    #     log.info("Getting PFO in Jet Frame")
    #     sample = get_PFO_in_JetFrame(sample)
    #     log.info("Removing Pile Up")
    #     sample = no_pile_up(sample)
    #     return sample

    root_dt = ROOTDataset.from_root_file(file)
    log.info('Loaded ROOT file.')
    root_dt.save(save_path)
    log.info('Saved dataset.')


# def test(saved_path: str) -> None:
#     root_dt = ROOTDataset.load(saved_path).dataset
#     print(f'Number of events: {root_dt.cardinality().numpy()}')
#     for i in root_dt.take(1):
#         print(i['jets_PFO_pt'])
#         print(i['jets_PFO_pt_JetFrame'])
#         print(i['jets_PartonTruthLabelID'])


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.save_path, exist_ok=True)
    root_to_preJIDENN_dataset(args.file_path, args.save_path)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
