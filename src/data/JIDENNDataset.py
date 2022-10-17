from dataclasses import dataclass
import uproot
import tensorflow as tf
from typing import Iterator
import numpy as np
import awkward as ak
import src.config.config_subclasses as cfg
# import pandas as pd
# import time
# import os


# #TODO: DELETE THIS
# @dataclass
# class Variables:
#     perJet: list[str]
#     perJetTuple: list[str]
#     perEvent: list[str]


# class cfg:
#     Variables = Variables


@dataclass
class JIDENNDataset:
    files: list[str]
    variables: cfg.Variables
    reading_size: int = 1000
    target: str | None = None
    weight: str | None = None
    num_workers: int | None = 4
    cut: str | None = None

    def __post_init__(self):
        self.jet_variables = [*self.variables.perJet]
        self.jet_variables += [*self.variables.perEvent] if self.variables.perEvent is not None else []

        self.jet_variables_target_weight = self.jet_variables + \
            [self.target] if self.target is not None else self.jet_variables
        self.jet_variables_target_weight += [self.weight] if self.weight is not None else []

        self.jet_tuple_variables = [*self.variables.perJetTuple] if self.variables.perJetTuple is not None else None

        self.all_variables = self.jet_variables + self.jet_tuple_variables if self.jet_tuple_variables is not None else self.jet_variables

        self.all_variables_target_weight = self.all_variables + \
            [self.target] if self.target is not None else self.all_variables
        self.all_variables_target_weight += [self.weight] if self.weight is not None else []

    def _data_iterator(self) -> Iterator[tuple[tuple[tf.Tensor, tf.RaggedTensor], tf.Tensor, tf.Tensor]]:
        for df in uproot.iterate(files=self.files,
                                 expressions=self.all_variables_target_weight,
                                 step_size=self.reading_size,
                                 cut=self.cut,
                                 num_workers=self.num_workers,
                                 file_handler=uproot.MultithreadedFileSource):  # type: ignore

            perJetTuple = ak.to_pandas(df[self.jet_tuple_variables])
            perJetTuple = tf.RaggedTensor.from_row_lengths(values=perJetTuple.values.astype(
                np.float32), row_lengths=perJetTuple.groupby(level=[0, 1]).size(), validate=True)

            perJet_target_weight = ak.to_pandas(df[self.jet_variables_target_weight])
            perJet = tf.constant(perJet_target_weight[self.jet_variables].values.astype(float), dtype=tf.float32)

            labels = perJet_target_weight[self.target].values.astype(int) if self.target is not None else None
            weight = perJet_target_weight[self.weight].values.astype(
                float) if self.weight is not None else tf.ones_like(labels, dtype=tf.float32)

            for sample in zip(perJet, perJetTuple, labels, weight):
                yield (sample[0], sample[1]), sample[2], sample[3]

    def _data_iterator_simple(self) -> Iterator[tuple[tuple[tf.Tensor, tf.RaggedTensor], tf.Tensor, tf.Tensor]]:
        for df in uproot.iterate(files=self.files,
                                 expressions=self.all_variables_target_weight,
                                 step_size=self.reading_size,
                                 cut=self.cut,
                                 num_workers=self.num_workers,
                                 file_handler=uproot.MultithreadedFileSource):  # type: ignore

            perJet_target_weight = ak.to_pandas(df[self.jet_variables_target_weight])
            perJet = tf.constant(perJet_target_weight[self.jet_variables].values.astype(float), dtype=tf.float32)

            labels = perJet_target_weight[self.target].values.astype(np.int32) if self.target is not None else None
            weight = perJet_target_weight[self.weight].values.astype(
                np.float32) if self.weight is not None else tf.ones_like(labels, dtype=tf.float32)

            yield perJet, labels, weight

    

    def get_preprocess_mapping(self):
        
        def parse_slice(slice_str: str):
            return tuple((slice(*(int(i) if i else None for i in part.strip().split(':'))) if ':' in part else int(part.strip())) for part in slice_str.split(','))
        
        @tf.function
        def _pick_var(sample, var):
            var_split = var.split('[')
            if len(var_split) == 1:
                return sample[var]
            var_name = var_split[0]
            var_slice = parse_slice(var_split[1][:-1]) if '[' in var else None
            if sample[var_name].shape.rank == 1:
                return sample[var_name][var_slice]
            return tf.squeeze(sample[var_name][var_slice], axis=-1) 

        @tf.function
        def pick_variables(sample):
            perJet = tf.stack([tf.cast(_pick_var(sample, var), tf.float32) for var in self.variables.perJet], axis=-1)

            perEvent = tf.stack([tf.cast(_pick_var(sample, var), tf.float32)
                                for var in self.variables.perEvent], axis=-1) if self.variables.perEvent is not None else None
            perEvent = tf.tile(perEvent[tf.newaxis, :], [tf.shape(perJet)[0], 1]) if perEvent is not None else None

            label = _pick_var(sample, self.target) if self.target is not None else None

            weight = _pick_var(sample, self.weight) if self.weight is not None else None
            weight = tf.fill([tf.shape(perJet)[0]], weight) if weight is not None else tf.ones_like(
                label, dtype=tf.float32)
            weight = weight*1e9

            if self.variables.perJetTuple is not None:
                perJetTuple = tf.stack([_pick_var(sample, var) for var in self.variables.perJetTuple], axis=-1)
                return (tf.concat([perJet, perEvent], axis=-1), perJetTuple), label, weight
            else:
                return tf.concat([perJet, perEvent], axis=-1), label, weight

        return pick_variables

    def load_single_dataset(self, file) -> tf.data.Dataset:
        dataset = tf.data.experimental.load(file, compression='GZIP', element_spec=self.dataset_TypeSpec)
        dataset = dataset.map(self.get_preprocess_mapping())
        return dataset

    @property
    def dataset(self):
        return tf.data.Dataset.from_tensor_slices(self.files).interleave(self.load_single_dataset, num_parallel_calls=tf.data.experimental.AUTOTUNE).flat_map(lambda *x: tf.data.Dataset.from_tensor_slices(x))

    @property
    def dataset_old(self) -> tf.data.Dataset:
        if self.jet_tuple_variables is not None:
            dt = tf.data.Dataset.from_generator(self._data_iterator,
                                                output_signature=(
                                                    (tf.TensorSpec(shape=(len(self.jet_variables), ), dtype=tf.float32),  # type:ignore
                                                     tf.TensorSpec(shape=(None, len(self.jet_tuple_variables)), dtype=tf.float32)),  # type:ignore
                                                    tf.TensorSpec(shape=(), dtype=tf.int32),  # type:ignore
                                                    tf.TensorSpec(shape=(), dtype=tf.float32),)  # type:ignore
                                                )
            # dt = dt.flat_map(lambda x,y,z,w : tf.data.Dataset.from_tensor_slices((x,y,z,w )))
            return dt
        else:
            dt = tf.data.Dataset.from_generator(self._data_iterator_simple,
                                                output_signature=(tf.TensorSpec(shape=(None, len(self.jet_variables), ), dtype=tf.float32),  # type:ignore
                                                                  # type:ignore
                                                                  tf.TensorSpec(shape=(None, ), dtype=tf.int32),
                                                                  tf.TensorSpec(shape=(None, ), dtype=tf.float32),)  # type:ignore
                                                )
            dt = dt.flat_map(lambda *x: tf.data.Dataset.from_tensor_slices((x)))
            return dt

    @property
    def dataset_TypeSpec(self):
        return {'GenFiltHT': tf.TensorSpec(shape=(), dtype=tf.float32, name=None),
                'HLT_j60': tf.TensorSpec(shape=(), dtype=tf.bool, name=None),
                'HLT_j45': tf.TensorSpec(shape=(), dtype=tf.bool, name=None),
                'jets_FracSamplingMaxIndex': tf.RaggedTensorSpec(tf.TensorShape([None]), tf.int32, 0, tf.int64),
                'jets_Width': tf.RaggedTensorSpec(tf.TensorShape([None]), tf.float32, 0, tf.int64),
                'HLT_j85': tf.TensorSpec(shape=(), dtype=tf.bool, name=None),
                'weight_mc': tf.RaggedTensorSpec(tf.TensorShape([None]), tf.float32, 0, tf.int64),
                'jets_JVFCorr': tf.RaggedTensorSpec(tf.TensorShape([None]), tf.float32, 0, tf.int64),
                'jets_ChargedPFOWidthPt1000': tf.RaggedTensorSpec(tf.TensorShape([None, None]), tf.float32, 1, tf.int64),
                'HLT_j15': tf.TensorSpec(shape=(), dtype=tf.bool, name=None),
                'corrected_averageInteractionsPerCrossing': tf.TensorSpec(shape=(), dtype=tf.float32, name=None),
                'jets_phi': tf.RaggedTensorSpec(tf.TensorShape([None]), tf.float32, 0, tf.int64),
                'HLT_j25': tf.TensorSpec(shape=(), dtype=tf.bool, name=None),
                'runNumber': tf.TensorSpec(shape=(), dtype=tf.uint64, name=None),
                'jets_JetConstitScaleMomentum_pt': tf.RaggedTensorSpec(tf.TensorShape([None]), tf.float32, 0, tf.int64),
                'jets_JetConstitScaleMomentum_eta': tf.RaggedTensorSpec(tf.TensorShape([None]), tf.float32, 0, tf.int64),
                'jets_NumChargedPFOPt1000': tf.RaggedTensorSpec(tf.TensorShape([None, None]), tf.int32, 1, tf.int64),
                'jets_SumPtChargedPFOPt500': tf.RaggedTensorSpec(tf.TensorShape([None, None]), tf.float32, 1, tf.int64),
                'eventNumber': tf.TensorSpec(shape=(), dtype=tf.uint64, name=None),
                'jets_ActiveArea4vec_phi': tf.RaggedTensorSpec(tf.TensorShape([None]), tf.float32, 0, tf.int64),
                'jets_EMFrac': tf.RaggedTensorSpec(tf.TensorShape([None]), tf.float32, 0, tf.int64),
                'HLT_j150': tf.TensorSpec(shape=(), dtype=tf.bool, name=None),
                'HLT_j380': tf.TensorSpec(shape=(), dtype=tf.bool, name=None),
                'jets_Timing': tf.RaggedTensorSpec(tf.TensorShape([None]), tf.float32, 0, tf.int64),
                'jets_fJVT': tf.RaggedTensorSpec(tf.TensorShape([None]), tf.float32, 0, tf.int64),
                'HLT_j110': tf.TensorSpec(shape=(), dtype=tf.bool, name=None),
                'jets_eta': tf.RaggedTensorSpec(tf.TensorShape([None]), tf.float32, 0, tf.int64),
                'jets_HadronConeExclExtendedTruthLabelID': tf.RaggedTensorSpec(tf.TensorShape([None]), tf.int32, 0, tf.int64),
                'jets_JetConstitScaleMomentum_phi': tf.RaggedTensorSpec(tf.TensorShape([None]), tf.float32, 0, tf.int64),
                'HLT_j35': tf.TensorSpec(shape=(), dtype=tf.bool, name=None),
                'jets_Jvt': tf.RaggedTensorSpec(tf.TensorShape([None]), tf.float32, 0, tf.int64),
                'lumiBlock': tf.TensorSpec(shape=(), dtype=tf.uint64, name=None),
                'sigma_v235': tf.TensorSpec(shape=(), dtype=tf.float32, name=None),
                'jets_NumChargedPFOPt500': tf.RaggedTensorSpec(tf.TensorShape([None, None]), tf.int32, 1, tf.int64),
                'jets_NumTrkPt1000': tf.RaggedTensorSpec(tf.TensorShape([None, None]), tf.int32, 1, tf.int64),
                'jets': tf.RaggedTensorSpec(tf.TensorShape([None]), tf.int32, 0, tf.int64),
                'weight_pu': tf.RaggedTensorSpec(tf.TensorShape([None]), tf.float32, 0, tf.int64),
                'jets_EnergyPerSampling': tf.RaggedTensorSpec(tf.TensorShape([None, None]), tf.float32, 1, tf.int64),
                'HLT_j55': tf.TensorSpec(shape=(), dtype=tf.bool, name=None),
                'HLT_j260': tf.TensorSpec(shape=(), dtype=tf.bool, name=None),
                'jets_m': tf.RaggedTensorSpec(tf.TensorShape([None]), tf.float32, 0, tf.int64),
                'jets_passFJVT': tf.RaggedTensorSpec(tf.TensorShape([None]), tf.bool, 0, tf.int64),
                'HLT_j420': tf.TensorSpec(shape=(), dtype=tf.bool, name=None),
                'jets_ActiveArea4vec_pt': tf.RaggedTensorSpec(tf.TensorShape([None]), tf.float32, 0, tf.int64),
                'sigma_v230': tf.TensorSpec(shape=(), dtype=tf.float32, name=None),
                'HLT_j360': tf.TensorSpec(shape=(), dtype=tf.bool, name=None),
                'jets_truth_partonPDG': tf.RaggedTensorSpec(tf.TensorShape([None]), tf.int32, 0, tf.int64),
                'jets_PartonTruthLabelID': tf.RaggedTensorSpec(tf.TensorShape([None]), tf.int32, 0, tf.int64),
                'jets_JetConstitScaleMomentum_m': tf.RaggedTensorSpec(tf.TensorShape([None]), tf.float32, 0, tf.int64),
                'jets_DetectorEta': tf.RaggedTensorSpec(tf.TensorShape([None]), tf.float32, 0, tf.int64),
                'jets_ActiveArea4vec_eta': tf.RaggedTensorSpec(tf.TensorShape([None]), tf.float32, 0, tf.int64),
                'jets_ConeTruthLabelID': tf.RaggedTensorSpec(tf.TensorShape([None]), tf.int32, 0, tf.int64),
                'jets_FracSamplingMax': tf.RaggedTensorSpec(tf.TensorShape([None]), tf.float32, 0, tf.int64),
                'JvtSF': tf.RaggedTensorSpec(tf.TensorShape([None]), tf.float32, 0, tf.int64),
                'jets_TrackWidthPt1000': tf.RaggedTensorSpec(tf.TensorShape([None, None]), tf.float32, 1, tf.int64),
                'jets_chf': tf.RaggedTensorSpec(tf.TensorShape([None]), tf.float32, 0, tf.int64),
                'HLT_j400': tf.TensorSpec(shape=(), dtype=tf.bool, name=None),
                'jets_passJVT': tf.RaggedTensorSpec(tf.TensorShape([None]), tf.bool, 0, tf.int64),
                'corrected_actualInteractionsPerCrossing': tf.TensorSpec(shape=(), dtype=tf.float32, name=None),
                'jets_HadronConeExclTruthLabelID': tf.RaggedTensorSpec(tf.TensorShape([None]), tf.int32, 0, tf.int64),
                'HLT_j0_perf_L1RD0_FILLED': tf.TensorSpec(shape=(), dtype=tf.bool, name=None),
                'metadata': tf.TensorSpec(shape=(3,), dtype=tf.float64, name=None),
                'jets_SumPtTrkPt500': tf.RaggedTensorSpec(tf.TensorShape([None, None]), tf.float32, 1, tf.int64),
                'jets_ActiveArea4vec_m': tf.RaggedTensorSpec(tf.TensorShape([None]), tf.float32, 0, tf.int64),
                'filtEff_v230': tf.TensorSpec(shape=(), dtype=tf.float32, name=None),
                'jets_n': tf.TensorSpec(shape=(), dtype=tf.int32, name=None),
                'jets_GhostMuonSegmentCount': tf.RaggedTensorSpec(tf.TensorShape([None]), tf.int32, 0, tf.int64),
                'mcChannelNumber': tf.TensorSpec(shape=(), dtype=tf.uint64, name=None),
                'FJvtSF': tf.RaggedTensorSpec(tf.TensorShape([None]), tf.float32, 0, tf.int64),
                'jets_JvtRpt': tf.RaggedTensorSpec(tf.TensorShape([None]), tf.float32, 0, tf.int64),
                'HLT_j175': tf.TensorSpec(shape=(), dtype=tf.bool, name=None),
                'jets_NumTrkPt500': tf.RaggedTensorSpec(tf.TensorShape([None, None]), tf.int32, 1, tf.int64),
                'jets_pt': tf.RaggedTensorSpec(tf.TensorShape([None]), tf.float32, 0, tf.int64),
                'filtEff_v235': tf.TensorSpec(shape=(), dtype=tf.float32, name=None)
                }


# def timing_decorator(func):
#     def wrapper(*args, **kwargs):
#         start = time.time()
#         result = func(*args, **kwargs)
#         end = time.time()
#         print(f'{func.__name__} took {end - start:.3f} seconds')
#         return result
#     return wrapper

# def setup():
#     dir = "/home/jankovys/JIDENN/data/data1/user.pleskot.mc16_13TeV.364707.JETM13.e7142_s3126_r10724_p4277.jetProp3_ANALYSIS/dev"
#     files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(".root")]
#     perJet = ["jets_ActiveArea4vec_eta", "jets_ActiveArea4vec_m", "jets_ActiveArea4vec_phi", "jets_ActiveArea4vec_pt", "jets_DetectorEta", "jets_EMFrac", "jets_FracSamplingMax", "jets_FracSamplingMaxIndex", "jets_GhostMuonSegmentCount", "jets_JVFCorr", "jets_JetConstitScaleMomentum_eta", "jets_JetConstitScaleMomentum_m", "jets_JetConstitScaleMomentum_phi", "jets_JetConstitScaleMomentum_pt", "jets_JvtRpt", "jets_Width", "jets_fJVT", "jets_passFJVT", "jets_passJVT", "jets_Jvt", "jets_Timing", "jets_chf", "jets_eta", "jets_m", "jets_phi", "jets_pt", "jets_ChargedPFOWidthPt1000[:,:,:1]"]
#     # perJetTuple = ["jets_ChargedPFOWidthPt1000", "jets_TrackWidthPt1000", "jets_NumChargedPFOPt1000", "jets_NumChargedPFOPt500", "jets_SumPtChargedPFOPt500", "jets_SumPtTrkPt500"]
#     perEvent = ["corrected_averageInteractionsPerCrossing"]
#     variables = cfg.Variables(
#         perJet=perJet,
#         perJetTuple=[],
#         perEvent=perEvent,
#         )
#     dataset = JIDENNDataset(files, variables, target="jets_truth_partonPDG", weight="weight_mc[:,:1]")
#     return dataset


# @timing_decorator
# def test(dataset, subset_len):

#     def test_func():
#         return dataset

#     for i, a in enumerate(test_func()):
#         print(f"{i+1}/{subset_len}: ", a[-1], end="\r")
#         if i == subset_len-1:
#             break
#     print()
#     print("done")


# if __name__ == '__main__':
#     # dir = "/home/jankovys/JIDENN/data/data1/user.pleskot.mc16_13TeV.364707.JETM13.e7142_s3126_r10724_p4277.jetProp3_ANALYSIS/dev"
#     # files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(".root")]
#     # files = files[:1]
#     perJet = ["jets_ActiveArea4vec_eta", "jets_ActiveArea4vec_m", "jets_ActiveArea4vec_phi", "jets_ActiveArea4vec_pt", "jets_DetectorEta", "jets_EMFrac", "jets_FracSamplingMax", "jets_FracSamplingMaxIndex", "jets_GhostMuonSegmentCount", "jets_JVFCorr", "jets_JetConstitScaleMomentum_eta",
#               "jets_JetConstitScaleMomentum_m", "jets_JetConstitScaleMomentum_phi", "jets_JetConstitScaleMomentum_pt", "jets_JvtRpt", "jets_Width", "jets_fJVT", "jets_passFJVT", "jets_passJVT", "jets_Jvt", "jets_Timing", "jets_chf", "jets_eta", "jets_m", "jets_phi", "jets_pt", "jets_ChargedPFOWidthPt1000[:,:1]"]
#     # perJetTuple = ["jets_ChargedPFOWidthPt1000", "jets_TrackWidthPt1000", "jets_NumChargedPFOPt1000", "jets_NumChargedPFOPt500", "jets_SumPtChargedPFOPt500", "jets_SumPtTrkPt500"]
#     perEvent = ["corrected_averageInteractionsPerCrossing"]
#     variables = cfg.Variables(
#         perJet = perJet,
#         perJetTuple = None,
#         perEvent = perEvent,
#     )
#     dir='/home/jankovys/JIDENN/data/dataset1/06'
#     files=[os.path.join(dir, f) for f in os.listdir(dir)]
#     dataset=JIDENNDataset(files, variables, target = "jets_truth_partonPDG", weight = "weight_mc[0]").datasetv3
#     # print(tf.data.DatasetSpec.from_value(dataset.load_single_dataset(files[0])))
#     for dato in dataset.take(1):
#         print(dato)

    # test(dataset.datasetv2, subset_len
    # test(dataset.dataset, subset_len)
