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
                                 cut=None,#self.cut, 
                                 num_workers=self.num_workers, 
                                 file_handler=uproot.MultithreadedFileSource)
    
    def _data_iterator(self) -> Iterator[tuple[tuple[tf.Tensor, tf.RaggedTensor], tf.Tensor, tf.Tensor]]:
        for df in self.root_iterator:  # type: ignore
            perJetTuple: pd.DataFrame = ak.to_pandas(df[self.jet_tuple_variables])
            perJet: pd.DataFrame = ak.to_pandas(df[self.jet_variables_target_weight])
            
            perJetTuple = perJetTuple.groupby(level=[0, 1]).agg(lambda x: x.tolist())
        
            df = pd.concat([perJet, perJetTuple], axis=1)
            df = df.query(self.cut) if self.cut is not None else df
            for _, row in df.iterrows():
                sample_data_perJet = tf.constant(row[self.jet_variables].values, dtype=tf.float32)
                sample_data_perJetTuple = tf.constant(row[self.jet_tuple_variables].to_list(), dtype=tf.float32)
                sample_data_perJetTuple = tf.transpose(sample_data_perJetTuple)
                sample_data_perJetTuple = tf.RaggedTensor.from_tensor(sample_data_perJetTuple, ragged_rank=1)
                sample_weight = row[self.weight] if self.weight is not None else 1.0
                sample_labels = row[self.target] if self.target is not None else None
                yield (sample_data_perJet, sample_data_perJetTuple), sample_labels, sample_weight
    
    def _data_iterator_simple(self) -> Iterator[tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
        for df in self.root_iterator:  # type: ignore
            df: pd.DataFrame = ak.to_pandas(df[self.jet_variables_target_weight])    
            df = df.query(self.cut) if self.cut is not None else df
            
            for _, row in df.iterrows():
                sample_data_perJet = tf.constant(row[self.jet_variables].values, dtype=tf.float32)
                sample_weight = row[self.weight] if self.weight is not None else 1.0
                sample_labels = row[self.target] if self.target is not None else None
                yield sample_data_perJet, sample_labels, sample_weight
                
            

    @property
    def dataset(self)->tf.data.Dataset:
        if self.jet_tuple_variables is not None:
            return tf.data.Dataset.from_generator(self._data_iterator,
                                                    output_signature=((tf.TensorSpec(shape=(len(self.jet_variables), ), dtype=tf.float32),        #type:ignore
                                                                        tf.RaggedTensorSpec(shape=(None, len(self.jet_tuple_variables)), dtype=tf.float32)),        #type:ignore
                                                                    tf.TensorSpec(shape=(), dtype=tf.int32),     #type:ignore
                                                                    tf.TensorSpec(shape=(), dtype=tf.float32),)      #type:ignore
                                                    )
        else:
            return tf.data.Dataset.from_generator(self._data_iterator_simple,
                                                    output_signature=(tf.TensorSpec(shape=(len(self.jet_variables), ), dtype=tf.float32),        #type:ignore
                                                                    tf.TensorSpec(shape=(), dtype=tf.int32),     #type:ignore
                                                                    tf.TensorSpec(shape=(), dtype=tf.float32),)      #type:ignore
                                                    )
            
        
    
        


