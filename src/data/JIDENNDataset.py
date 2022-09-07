from dataclasses import dataclass
import uproot
import tensorflow as tf
from typing import Iterator, Optional, List, Tuple

# /work/sched3am/exTau/jets1tau-v01/nom/user.scheiric.mc16_13TeV.41047*

    
@dataclass
class JIDENNDataset:
    files: List[str]
    variables: List[str]
    reading_size : int = 2_048
    target: Optional[str] = None
    weight : Optional[str] = None
    num_workers: Optional[int] = 1
    cut: Optional[str] = None # 40 < tau_pt < 60
    
    
    def _data_iterator_pd(self) -> Iterator[Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
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
    
            

    @property
    def dataset(self)->tf.data.Dataset:
        dataset = tf.data.Dataset.from_generator(self._data_iterator_pd,
                                            output_signature=(tf.TensorSpec(shape=(None, len(self.variables)), dtype=tf.float32),  # type: ignore
                                                                tf.TensorSpec(shape=(None, ), dtype=tf.int32), # type: ignore
                                                                tf.TensorSpec(shape=(None, ), dtype=tf.float32)))  # type: ignore
        
        dataset = dataset.flat_map(lambda x, y, z: tf.data.Dataset.from_tensor_slices((x, y, z)))
        return dataset
    
        


    

    
