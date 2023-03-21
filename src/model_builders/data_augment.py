import tensorflow as tf
import numpy as np
from typing import Optional, Tuple, Union


class InfraredAugmentation:
    def __init__(self) -> None:
        pass
    
    def drop_soft_constit(self, drop_prob: float, jet_constits: tf.Tensor) -> tf.Tensor:
        """Drop soft constituents from a jet.
        
        Args:
            drop_prob: Probability of dropping a constituent.
            jet_constits: Tensor of shape (N, C) of jet constituents. Where N is the number of constituents and C is the number of features.
        
        Returns:
            jet_constits: Tensor of shape (N, C) of jet constituents.
        """
        # Drop constituents
        drop_mask = tf.random.uniform(shape=tf.shape(jet_constits)[0]) > drop_prob
        jet_constits = tf.boolean_mask(jet_constits, drop_mask)
        
        return jet_constits