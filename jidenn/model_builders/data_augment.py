import tensorflow as tf
import numpy as np
from typing import Optional, Tuple, Union


class InfraredAugmentation:
    def __init__(self, drop_prob: float, rise: float = 1.):
        self.drop_prob = drop_prob
        self.rise = rise

    def drop_soft_constit(self, jet_constits: tf.Tensor, drop_mask) -> Union[tf.Tensor, tf.RaggedTensor]:
        """Drop soft constituents from a jet.

        Args:
            drop_prob: Probability of dropping a constituent.
            jet_constits: Tensor of shape (N, ) of jet constituents. Where N is the number of constituents.

        Returns:
            jet_constits: Tensor of shape (N, ) of jet constituents.
        """
        # Drop constituents
        jet_constits = tf.ragged.boolean_mask(jet_constits, drop_mask)
        return jet_constits

    def __call__(self, sample: dict) -> dict:
        """Apply data augmentation to a sample.

        Args:
            sample: Dictionary containing the sample data.

        Returns:
            sample: Dictionary containing the sample data.
        """
        new_sample = {}
        # Drop constituents
        # weight_drop = tf.cast(mask, tf.float32)
        weight_drop = tf.random.uniform(shape=tf.shape(sample[list(sample.keys())[0]]), minval=0, maxval=1)
        # drop_mask = drop_mask * tf.nn.sigmoid(-sample['log_PT|PTjet']) > self.drop_prob
        drop_prob = (tf.exp((1 - tf.exp(self.rise * sample['log_PT|PTjet']))) - 1) / (tf.exp(1.) - 1)
        drop_mask = weight_drop > drop_prob * self.drop_prob
        for var in sample:
            new_sample[var] = self.drop_soft_constit(sample[var], drop_mask)
        new_sample['mask'] = self.drop_soft_constit(drop_prob, drop_mask)
        # new_sample['mask'] = drop_mask
        return new_sample
