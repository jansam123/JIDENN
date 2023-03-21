import tensorflow as tf
import numpy as np
from typing import Optional, Tuple, Union


class InfraredAugmentation:
    def __init__(self, drop_prob: float):
        self.drop_prob = drop_prob

    def drop_soft_constit(self, jet_constits: tf.Tensor) -> tf.Tensor:
        """Drop soft constituents from a jet.

        Args:
            drop_prob: Probability of dropping a constituent.
            jet_constits: Tensor of shape (N, C) of jet constituents. Where N is the number of constituents and C is the number of features.

        Returns:
            jet_constits: Tensor of shape (N, C) of jet constituents.
        """
        # Drop constituents
        drop_mask = tf.random.uniform(shape=tf.shape(jet_constits)[0]) > self.drop_prob
        jet_constits = tf.boolean_mask(jet_constits, drop_mask)

        return jet_constits

    def __call__(self, sample: dict) -> dict:
        """Apply data augmentation to a sample.

        Args:
            sample: Dictionary containing the sample data.

        Returns:
            sample: Dictionary containing the sample data.
        """
        # Drop constituents
        for var in sample:
            sample[var] = self.drop_soft_constit(sample[var])

        return sample
