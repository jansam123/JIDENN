from dataclasses import dataclass
from unittest.mock import PropertyMock
import uproot
import tensorflow as tf
from typing import Generator, Iterator, Optional
from utils import timing_decorator

# /work/sched3am/exTau/jets1tau-v01/nom
# ls user.scheiric.mc16_13TeV.41047*



class JIDENNDataset:
    target: str = 'jets_truth_partonPDG' # matchJetPdgId
    variables: list[str] = ['jets_Jvt', 'jets_Timing', 'jets_chf', 'jets_eta', 'jets_fmax',
                             'jets_m', 'jets_phi', 'jets_pt', 'jets_sumPtTrk']
    #taus_TrackWidthPt500TV, taus_seed*, 
    target_mapping: dict[int, int] = {-1: 0, # unknown
                                      1: 1, # u quark  
                                      2: 2, # d quark
                                      3: 3, # s quark
                                      4: 4, # c quark
                                      5: 5, # b quark
                                      21: 6 # gluon
                                      }
    LABELS: int = len(target_mapping)
    # SIZE: int = 102_126
        
    
    @dataclass
    class Dataset:
        files: list[str]
        target: str
        variables: list[str]
        reading_size : Optional[int] = 1000
        cut: Optional[str] = None # 40 < tau_pt < 60
        save: Optional[str] = None
        load: Optional[str] = None
        
        
        def _data_iterator(self) -> Iterator[tuple[tf.Tensor, tf.Tensor]]:
            for df in uproot.iterate(files=self.files, expressions=self.variables+[self.target],  step_size=self.reading_size, library='pd', cut=self.cut):  # type: ignore
                sample_data = df[self.variables]
                sample_data = tf.convert_to_tensor(sample_data)
                sample_data = tf.reshape(sample_data, [-1, len(self.variables)])
                
                sample_labels = df[self.target].map(JIDENNDataset.target_mapping)
                sample_labels = tf.convert_to_tensor(sample_labels)
                sample_labels = tf.reshape(sample_labels, [-1,])
                
                yield sample_data, sample_labels
            
        @property
        def dataset(self)->tf.data.Dataset:
            dataset = tf.data.Dataset.from_generator(self._data_iterator,
                                                output_signature=(tf.TensorSpec(shape=(None, len(self.variables)), dtype=tf.float32),  # type: ignore
                                                                    tf.TensorSpec(shape=(None, ), dtype=tf.int32)))  # type: ignore
            dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
            dataset = dataset.apply(tf.data.experimental.assert_cardinality(sum(1 for _ in dataset)))
            return dataset
        
        def _dataset_to_tfrecord(self, path)->None:
            with tf.io.TFRecordWriter(path) as file_writer:
                for sample_data, sample_labels in self.dataset:
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'data': tf.train.Feature(float_list=tf.train.FloatList(value=sample_data.numpy().flatten())),
                        'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=sample_labels.numpy().flatten()))
                    }))
                    file_writer.write(example.SerializeToString())  # type: ignore
                        
            
        def _tfrecord_to_dataset(self, path:str) -> tf.data.Dataset:
            def parse_tfrecord_fn(example):
                feature_description = {
                    "data": tf.io.VarLenFeature(tf.float32),
                    "label": tf.io.FixedLenFeature([], tf.int64),
                }
                example = tf.io.parse_single_example(example, feature_description)
                example["data"] = tf.sparse.to_dense(example["data"])
                return example["data"], example["label"]
            return tf.data.TFRecordDataset(path).map(parse_tfrecord_fn)
            
                


        
    train: Dataset
    # dev: Dataset

    def __init__(self, files: list[str], save: Optional[str]=None, load: Optional[str]=None, reading_size:Optional[int]=1000, cut: Optional[str] = None) -> None:
        self.train = self.Dataset(files, self.target, self.variables, reading_size, save=save, load=load, cut=cut)
    

    
