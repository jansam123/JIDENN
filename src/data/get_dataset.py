from .JIDENNDataset import JIDENNDataset
from src.config import config_subclasses as cfg        
import tensorflow as tf
        
def get_gluon_dataset(args_data:cfg.Data, 
                      files:list[str],) -> tf.data.Dataset:
    

    
    dataset = JIDENNDataset(files=files,
                         variables=args_data.variables,
                         target=args_data.target,
                         weight=args_data.weight,
                         reading_size=args_data.reading_size,
                         num_workers=1,
                         cut=args_data.cut,).dataset
    
    dataset = dataset.filter(lambda x,y,z: tf.math.equal(y, args_data.raw_gluon))
    return dataset
                         
                         
def get_quark_dataset(args_data:cfg.Data, 
                      files:list[str],) -> tf.data.Dataset:
    
    

    dataset =  JIDENNDataset(files=files,
                         variables=args_data.variables,
                         target=args_data.target,
                         weight=args_data.weight,
                         reading_size=args_data.reading_size,
                         num_workers=1,
                         cut=args_data.cut,).dataset
    quark_labels = tf.constant(args_data.raw_quarks)
    dataset = dataset.filter(lambda x,y,z:tf.math.reduce_any(tf.math.equal(y, quark_labels)))
    return dataset

def get_mixed_dataset(args_data:cfg.Data, 
                      files:list[str],) -> tf.data.Dataset:
    

    
    dataset =  JIDENNDataset(files=files,
                         variables=args_data.variables,
                         target=args_data.target,
                         weight=args_data.weight,
                         reading_size=args_data.reading_size,
                         num_workers=1,
                         cut=args_data.cut,).dataset
    dataset = dataset.filter(lambda x,y,z: tf.math.reduce_any(y in args_data.raw_quarks or y == args_data.raw_gluon))
    return dataset
                         
                         
                         
