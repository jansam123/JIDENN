from dataclasses import dataclass
import uproot
import pandas as pd
import tensorflow as tf
from typing import Iterator
import timeit
import numpy as np
import awkward as ak
from src.config.config_subclasses import Variables

# /work/sched3am/exTau/jets1tau-v01/nom/user.scheiric.mc16_13TeV.41047*


@dataclass
class JIDENNDataset:
    files: list[str]
    variables: Variables
    reading_size : int = 2_048
    target: str | None = None
    weight : str | None = None
    num_workers: int | None = 1
    cut: str | None = None
    
    
    def __post_init__(self):
        self.all_variables = [*self.variables.perEvent, *self.variables.perJet, *self.variables.perJetTuple]
        self.jet_variables = [*self.variables.perJet, *self.variables.perEvent]
        self.jet_variables_target_weight = [*self.variables.perJet, *self.variables.perEvent, self.target, self.weight]
        self.jet_tuple_variables = [*self.variables.perJetTuple]
    
    
    def _data_iterator(self) -> Iterator[tuple[tf.RaggedTensor, tf.Tensor, tf.Tensor]]:
        target = [self.target] if self.target is not None else []
        weight = [self.weight] if self.weight is not None else []
        expressions = target + weight
        expressions += [*self.variables.perEvent]
        expressions += [*self.variables.perJet]
        expressions += [*self.variables.perJetTuple]
        
        for df in uproot.iterate(files=self.files, 
                                 expressions=expressions,  
                                 step_size=self.reading_size, 
                                 cut=None,#self.cut, 
                                 num_workers=self.num_workers, 
                                 file_handler=uproot.MultithreadedFileSource):  # type: ignore
            # cut = ak.any(df['jets_truth_partonPDG'] > 0, axis=1)
            
            # perJet = ak.to_pandas(df[self.variables.perJet])
            perJetTuple: pd.DataFrame = ak.to_pandas(df[self.jet_tuple_variables])
            perJet: pd.DataFrame = ak.to_pandas(df[self.jet_variables_target_weight])
            
            perJetTuple = perJetTuple.groupby(level=[0, 1]).agg(lambda x: x.tolist())
        
            df = pd.concat([perJet, perJetTuple], axis=1)
            df = df.query(self.cut) if self.cut is not None else df
            
            for _, row in df.iterrows():
                sample_data_perJet = tf.ragged.constant(row[self.jet_variables].values, dtype=tf.float32)[:, tf.newaxis]
                sample_data_perJetTuple = tf.ragged.constant(row[self.jet_tuple_variables])
                sample_data = tf.concat([sample_data_perJet, sample_data_perJetTuple], axis=0)
                sample_weight = row[self.weight] if self.weight is not None else 1.0
                sample_labels = row[self.target] if self.target is not None else None
                yield sample_data, sample_labels, sample_weight
            
                
            

    @property
    def dataset(self)->tf.data.Dataset:
        dataset = tf.data.Dataset.from_generator(self._data_iterator,
                                                 output_signature=(tf.RaggedTensorSpec(shape=(len(self.all_variables), None), dtype=tf.float32),        #type:ignore
                                                                   tf.TensorSpec(shape=(), dtype=tf.int32),     #type:ignore
                                                                   tf.TensorSpec(shape=(), dtype=tf.float32),)      #type:ignore
                                                 )
        return dataset
    
        


