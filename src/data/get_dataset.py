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
                         num_workers=6,
                         cut=args_data.cut,).dataset
    
    @tf.function
    def filter_gluons(x,y,z):
        return y == args_data.raw_gluon
    
    dataset = dataset.filter(filter_gluons)
    return dataset
                         
                         
def get_quark_dataset(args_data:cfg.Data, 
                      files:list[str],) -> tf.data.Dataset:

    dataset =  JIDENNDataset(files=files,
                         variables=args_data.variables,
                         target=args_data.target,
                         weight=args_data.weight,
                         reading_size=args_data.reading_size,
                         num_workers=6,
                         cut=args_data.cut,).dataset
    quark_labels = tf.constant(args_data.raw_quarks)

    @tf.function
    def filter_quarks(x,y,z):
        return tf.reduce_any(tf.equal(y, quark_labels))
    
    dataset = dataset.filter(filter_quarks)
    return dataset

def get_mixed_dataset(args_data:cfg.Data, 
                      files:list[str],) -> tf.data.Dataset:
    
    dataset =  JIDENNDataset(files=files,
                         variables=args_data.variables,
                         target=args_data.target,
                         weight=args_data.weight,
                         reading_size=args_data.reading_size,
                         num_workers=6,
                         cut=args_data.cut,).dataset
    
    @tf.function
    def filter_mixed(x,y,z):
        return y == args_data.raw_gluon or y in args_data.raw_quarks
    
    dataset = dataset.filter(filter_mixed)
    return dataset
                         
                         
                         
