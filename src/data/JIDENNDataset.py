from dataclasses import dataclass
import uproot
import pandas as pd
import tensorflow as tf
from typing import Iterator
import time
import numpy as np
import awkward as ak
import sys
# from src.config.config_subclasses import Variables


#TODO: DELETE THIS 
@dataclass
class Variables:
    perJet: list[str]
    perJetTuple: list[str]
    perEvent: list[str]




@dataclass
class JIDENNDataset:
    files: list[str]
    variables: Variables
    reading_size : int = 1000
    target: str | None = None
    weight : str | None = None
    num_workers: int | None = 4
    cut: str | None = None
    
    
    def __post_init__(self):
        self.jet_variables = [*self.variables.perJet]
        self.jet_variables += [*self.variables.perEvent] if self.variables.perEvent is not None else []
        
        self.jet_variables_target_weight = self.jet_variables + [self.target] if self.target is not None else self.jet_variables
        self.jet_variables_target_weight += [self.weight] if self.weight is not None else []
        
        self.jet_tuple_variables = [*self.variables.perJetTuple] if self.variables.perJetTuple is not None else None

        self.all_variables = self.jet_variables + self.jet_tuple_variables if self.jet_tuple_variables is not None else self.jet_variables
        
        self.all_variables_target_weight = self.all_variables + [self.target] if self.target is not None else self.all_variables
        self.all_variables_target_weight += [self.weight] if self.weight is not None else []
        
        self.root_iterator = uproot.iterate(files=self.files, 
                                 expressions=self.all_variables_target_weight,  
                                 step_size=self.reading_size, 
                                 cut=self.cut, 
                                 num_workers=self.num_workers, 
                                 file_handler=uproot.MultithreadedFileSource)
        
    
    def _data_iterator(self) -> Iterator[tuple[tuple[tf.Tensor, tf.RaggedTensor], tf.Tensor, tf.Tensor]]:
        for df in self.root_iterator:  # type: ignore
            perJetTuple: pd.DataFrame = ak.to_pandas(df[self.jet_tuple_variables])
            perJet: pd.DataFrame = ak.to_pandas(df[self.jet_variables_target_weight])
            
            perJetTuple = perJetTuple.groupby(level=[0, 1]).agg(lambda x: x.tolist())

            
            df = pd.concat([perJet, perJetTuple], axis=1)
            df = df.query(self.cut) if self.cut is not None else df
            
            data_perJet = tf.constant(df[self.jet_variables], dtype=tf.float32)
            weight = tf.constant(df[self.weight], dtype=tf.float32) if self.weight is not None else 1.0
            labels = tf.constant(df[self.target], dtype=tf.int32) if self.target is not None else None
            
            
            to_stack = []
            for ragged_var in self.variables.perJetTuple:
                var = df[ragged_var].reset_index(drop=True).explode()
                var = tf.RaggedTensor.from_value_rowids(values=var.values.astype(float), value_rowids=var.index.values, validate=False)
                to_stack.append(var)
                
            data_perJetTuple = tf.stack(to_stack, axis=-1)
            
            # for sample in zip(data_perJet, data_perJetTuple, labels, weight):
            #     yield (sample[0]), sample[2], sample[3]
        
    def _data_iterator_v2(self) -> Iterator[tuple[tuple[tf.Tensor, tf.RaggedTensor], tf.Tensor, tf.Tensor]]:
        for df in self.root_iterator:  # type: ignore
            perJetTuple = ak.to_pandas(df[self.jet_tuple_variables])
            perJetTuple = tf.RaggedTensor.from_row_lengths(values=perJetTuple.values.astype(np.float32), row_lengths=perJetTuple.groupby(level=[0, 1]).size(), validate=True)
            
            perJet_target_weight = ak.to_pandas(df[self.jet_variables_target_weight])
            perJet = tf.constant(perJet_target_weight[self.jet_variables].values.astype(float), dtype=tf.float32)
            
            weight = perJet_target_weight[self.weight].values.astype(float) if self.weight is not None else None
            labels = perJet_target_weight[self.target].values.astype(int) if self.target is not None else None
            
            for sample in zip(perJet, perJetTuple, labels, weight):
                yield (sample[0], sample[1]), sample[2], sample[3]
            
    
    def _data_iterator_simple(self) -> Iterator[tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
        for df in self.root_iterator:  # type: ignore
            df: pd.DataFrame = ak.to_pandas(df[self.jet_variables_target_weight])    
            df = df.query(self.cut) if self.cut is not None else df
            
            for _, row in df.iterrows():
                data_perJet = tf.constant(row[self.jet_variables].values, dtype=tf.float32)
                weight = row[self.weight] if self.weight is not None else 1.0
                labels = row[self.target] if self.target is not None else None
                yield data_perJet, labels, weight
                
            

    @property
    def dataset(self)->tf.data.Dataset:
        if self.jet_tuple_variables is not None:
            dt = tf.data.Dataset.from_generator(self._data_iterator_v2,
                                                    output_signature=(
                                                                    (tf.TensorSpec(shape=(len(self.jet_variables), ), dtype=tf.float32),        #type:ignore
                                                                    tf.TensorSpec(shape=(None, len(self.jet_tuple_variables)), dtype=tf.float32)),        #type:ignore
                                                                    tf.TensorSpec(shape=(), dtype=tf.int32),     #type:ignore
                                                                    tf.TensorSpec(shape=(), dtype=tf.float32),)      #type:ignore
                                                    )
            # dt = dt.flat_map(lambda x,y,z,w : tf.data.Dataset.from_tensor_slices((x,y,z,w )))
            return dt
        else:
            return tf.data.Dataset.from_generator(self._data_iterator_simple,
                                                    output_signature=(tf.TensorSpec(shape=(len(self.jet_variables), ), dtype=tf.float32),        #type:ignore
                                                                    tf.TensorSpec(shape=(), dtype=tf.int32),     #type:ignore
                                                                    tf.TensorSpec(shape=(), dtype=tf.float32),)      #type:ignore
                                                    )
            
        
    
        
        
        
# def timing_decorator(func):
#     def wrapper(*args, **kwargs):
#         start = time.time()
#         result = func(*args, **kwargs)
#         end = time.time()
#         print(f'{func.__name__} took {end - start:.3f} seconds')
#         return result
#     return wrapper

# def setup():
#     files = ["/home/home-pc/bakalarka/JIDENN/data/data1/user.pleskot.mc16_13TeV.364703.JETM13.e7142_s3126_r10724_p4277.jetProp3_ANALYSIS/user.pleskot.30561067.ANALYSIS._000020.root"]
#     perJet = ["jets_ActiveArea4vec_eta", "jets_ActiveArea4vec_m", "jets_ActiveArea4vec_phi", "jets_ActiveArea4vec_pt", "jets_DetectorEta", "jets_EMFrac", "jets_FracSamplingMax", "jets_FracSamplingMaxIndex", "jets_GhostMuonSegmentCount", "jets_JVFCorr", "jets_JetConstitScaleMomentum_eta", "jets_JetConstitScaleMomentum_m", "jets_JetConstitScaleMomentum_phi", "jets_JetConstitScaleMomentum_pt", "jets_JvtRpt", "jets_Width", "jets_fJVT", "jets_passFJVT", "jets_passJVT", "jets_Jvt", "jets_Timing", "jets_chf", "jets_eta", "jets_m", "jets_phi", "jets_pt"]
#     perJetTuple = ["jets_ChargedPFOWidthPt1000", "jets_TrackWidthPt1000", "jets_NumChargedPFOPt1000", "jets_NumChargedPFOPt500", "jets_SumPtChargedPFOPt500", "jets_SumPtTrkPt500"]
#     perEvent = ["corrected_averageInteractionsPerCrossing"]
#     variables = Variables(
#         perJet=perJet,
#         perJetTuple=perJetTuple,
#         perEvent=perEvent,
#         )
#     dataset = JIDENNDataset(files, variables, target="jets_truth_partonPDG", weight="weight_mc[:,0]", cut="jets_truth_partonPDG>0")
#     return dataset


# @timing_decorator
# def test(dataset, subset_len):
#     for idx, (x,y,z) in enumerate(dataset.take(subset_len)):
#         print(f"{idx+1}/{subset_len}: ", x[1].shape, y.shape, z.shape, end="\r")

        
        
#     print()
#     print("done")
        
# if __name__ == '__main__':
#     dataset = setup()
#     subset_len = 10000    
#     test(dataset.dataset, subset_len)

