from dataclasses import dataclass
import uproot
# import pandas as pd
import tensorflow as tf
from typing import Iterator
import timeit
import numpy as np

# /work/sched3am/exTau/jets1tau-v01/nom/user.scheiric.mc16_13TeV.41047*


@dataclass
class JIDENNDataset:
    files: list[str]
    variables: list[str]
    reading_size : int = 2_048
    target: str | None = None
    weight : str | None = None
    num_workers: int | None = 1
    cut: str | None = None 
    
    
    def _data_iterator(self) -> Iterator[tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
        target =[self.target] if self.target is not None else []
        weight =[self.weight] if self.weight is not None else []
        expressions = self.variables + target + weight
        
        for df in uproot.iterate(files=self.files, 
                                 expressions=expressions,  
                                 step_size=self.reading_size, 
                                 cut=self.cut, 
                                 num_workers=self.num_workers, 
                                 file_handler=uproot.MultithreadedFileSource, 
                                 library='pd'):  # type: ignore
            
            sample_data = df[self.variables]
            sample_data = tf.convert_to_tensor(sample_data, dtype=tf.float64)
            sample_data = tf.reshape(sample_data, [-1, len(self.variables)])
            
                
            sample_labels = df[self.target]
            sample_labels = tf.convert_to_tensor(sample_labels, dtype=tf.int32)
            sample_labels = tf.reshape(sample_labels, [-1,])
            
            if self.weight is None:
                yield sample_data, sample_labels, tf.ones_like(sample_labels)
                
            else:
                sample_weight = df[self.weight]
                sample_weight = tf.convert_to_tensor(sample_weight, dtype=tf.float64)
                sample_weight = tf.reshape(sample_weight, [-1,])
                
                yield sample_data, sample_labels, sample_weight
                
                
    def _data_iteratorv2(self) -> Iterator[tuple[tf.Tensor, tf.Tensor | None, tf.Tensor]]:
        target =[self.target] if self.target is not None else []
        weight =[self.weight] if self.weight is not None else []
        expressions = self.variables + target + weight
        
        
        for df in uproot.iterate(files=self.files, 
                                 expressions=expressions,  
                                 step_size=self.reading_size, 
                                 cut=self.cut, 
                                 num_workers=self.num_workers, 
                                 file_handler=uproot.MultithreadedFileSource, 
                                 library='pd'):  # type: ignore
            
            
            lenghts = list(df.groupby(level=0).count()[self.variables[0]])
            data = tf.RaggedTensor.from_row_lengths(df, row_lengths=lenghts)
            
            data = data.merge_dims(0,1)
            
            sample_weight = data[:,-1] if self.weight is not None else tf.ones_like(data[:,0])
            sample_labels = tf.cast(data[:,-2], tf.int32) if self.target is not None else None
            sample_data = data[:,:-2]
            

            yield sample_data, sample_labels, sample_weight
            

            

    @property
    def dataset(self)->tf.data.Dataset:
        dataset = tf.data.Dataset.from_generator(self._data_iterator,
                                            output_signature=(tf.TensorSpec(shape=(None, len(self.variables)), dtype=tf.float32),  # type: ignore
                                                                tf.TensorSpec(shape=(None, ), dtype=tf.int32), # type: ignore
                                                                tf.TensorSpec(shape=(None, ), dtype=tf.float32)))  # type: ignore
        
        dataset = dataset.flat_map(lambda x, y, z: tf.data.Dataset.from_tensor_slices((x, y, z)))
        return dataset
    
        



