from dataclasses import dataclass
import uproot
import tensorflow as tf
from typing import Iterator, Optional
from functools import partial
import pandas as pd

# /work/sched3am/exTau/jets1tau-v01/nom/user.scheiric.mc16_13TeV.41047*




class JIDENNDataset:
    target: str = 'jets_truth_partonPDG'  #'taus_truth_matchJetPdgId' 
    variables: list[str] = ['jets_Jvt', 'jets_Timing', 'jets_chf', 'jets_eta', 'jets_fmax',
                            'jets_m', 'jets_phi', 'jets_pt', 'jets_sumPtTrk']
                            # [ 'taus_seedJetE', 'taus_seedJetEta', 'taus_seedJetPhi',
                            # 'taus_seedJetPt',  'taus_dRminToJet', 'taus_TrackWidthPt500TV', 'taus_seedJetWidth']
    #taus_TrackWidthPt500TV, taus_seed*, 
    weight : Optional[str] = 'weight_mc[:,0]'
    target_mapping: dict[int, int] = {-1: -1, # unknown
                                      -999: -1, # unknown
                                      1: 1, # u quark  
                                      2: 1, # d quark
                                      3: 1, # s quark
                                      4: 1, # c quark
                                      5: 1, # b quark
                                      21: 0 # gluon
                                      }
    LABELS: int = 2
    # SIZE: int = 102_126
        
    
    @dataclass
    class Dataset:
        files: list[str]
        variables: list[str]
        target: Optional[str] = None
        weight : Optional[str] = None
        reading_size : Optional[int] = 2_048
        num_workers: Optional[int] = 1
        cut: Optional[str] = None # 40 < tau_pt < 60
        save: Optional[str] = None
        load: Optional[str] = None
        
        
        def _data_iterator_pd(self, label_subset:Optional[int]=None) -> Iterator[tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
            target =[self.target] if self.target is not None else []
            weight =[self.weight] if self.weight is not None else []
            expressions = self.variables  + target + weight
            for df in uproot.iterate(files=self.files, expressions=expressions,  step_size=self.reading_size, library='pd', cut=self.cut, num_workers=self.num_workers, file_handler=uproot.MultithreadedFileSource):  # type: ignore
                df[self.target] = df[self.target].map(JIDENNDataset.target_mapping)
                df = df[df[self.target] != -1]
                # with pd.option_context('display.max_rows', None): 
                #     print(df)
                if label_subset is not None:
                    df = df[df[self.target]==label_subset]
                
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
                
        
        def _data_iterator(self) -> Iterator[tuple[tf.Tensor, tf.Tensor]]:
            for arr in uproot.iterate(files=self.files, expressions=self.variables+[self.target],  step_size=self.reading_size, library='np', cut=self.cut, num_workers=self.num_workers, file_handler=uproot.MultithreadedFileSource):  # type: ignore
                sample_labels = tf.ragged.constant(arr[self.target], dtype=tf.int64).merge_dims(0, -1)
                unknown_mask = tf.equal(sample_labels, -1)
                sample_labels = tf.boolean_mask(sample_labels, tf.logical_not(unknown_mask))
                sample_labels = self.lookup_table.lookup(sample_labels)
                sample_data = tf.stack([tf.ragged.constant(arr[var]).merge_dims(0, -1) for var in self.variables], axis=1)
                sample_data = tf.boolean_mask(sample_data, tf.logical_not(unknown_mask))
                yield sample_data, sample_labels
                
        @property    
        def lookup_table(self) -> tf.lookup.StaticVocabularyTable:
            return tf.lookup.StaticVocabularyTable(
                tf.lookup.KeyValueTensorInitializer(
                    list(JIDENNDataset.target_mapping.keys()),
                    list(JIDENNDataset.target_mapping.values()),
                    key_dtype=tf.int64,
                    value_dtype=tf.int64,
                ),
                num_oov_buckets=1,
                )
            
        @property
        def datasets(self)->list[tf.data.Dataset]:
            datasets = []
            for label in [0,1]:
                dataset = tf.data.Dataset.from_generator(partial(self._data_iterator_pd, label),
                                                    output_signature=(tf.TensorSpec(shape=(None, len(self.variables)), dtype=tf.float32),  # type: ignore
                                                                        tf.TensorSpec(shape=(None, ), dtype=tf.int32), # type: ignore
                                                                        tf.TensorSpec(shape=(None, ), dtype=tf.float32)))  # type: ignore
                
                dataset = dataset.flat_map(lambda x, y, z: tf.data.Dataset.from_tensor_slices((x, y, z)))
                datasets.append(dataset)
            return datasets

        @property
        def dataset(self)->tf.data.Dataset:
            dataset = tf.data.Dataset.from_generator(self._data_iterator_pd,
                                                output_signature=(tf.TensorSpec(shape=(None, len(self.variables)), dtype=tf.float32),  # type: ignore
                                                                    tf.TensorSpec(shape=(None, ), dtype=tf.int32), # type: ignore
                                                                    tf.TensorSpec(shape=(None, ), dtype=tf.float32)))  # type: ignore
            
            dataset = dataset.flat_map(lambda x, y, z: tf.data.Dataset.from_tensor_slices((x, y, z)))
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
    dev: Dataset
    test: Dataset

    def __init__(self, files: list[str], 
                 dev_size: float=0,
                 test_size: float=0,
                 save: Optional[str]=None, 
                 load: Optional[str]=None, 
                 reading_size:Optional[int]=1000, 
                 cut: Optional[str]=None, 
                 num_workers: Optional[int]=1) -> None:
        
        num_files = len(files)
        num_dev_files = int(num_files * dev_size)
        num_test_files = int(num_files * test_size)
        
        if num_dev_files >= 1 and dev_size != 0:
            self.dev = self.Dataset(files[:num_dev_files+1], self.variables, self.target, self.weight, reading_size, save=save, load=load, cut=cut, num_workers=num_workers)
            files = files[num_dev_files+1:]
            
        if num_test_files >= 1 and test_size != 0:
            self.test = self.Dataset(files[:num_test_files+1], self.variables, self.target, self.weight, reading_size, save=save, load=load, cut=cut, num_workers=num_workers)
            files = files[num_test_files+1:]
            
        self.train = self.Dataset(files, self.variables, self.target, self.weight, reading_size, save=save, load=load, cut=cut, num_workers=num_workers)

    

    
