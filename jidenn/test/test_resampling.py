import pytest
import tempfile
import os
import numpy as np
import tensorflow as tf
import logging
import pandas as pd
logging.basicConfig(level=logging.INFO)

from jidenn.preprocess.resampling import compute_uniform_weights
from jidenn.data.JIDENNDataset import all_equal


def test_compute_uniform_weights():
    bin_counts = tf.constant([100, 200, 300, 400, 500])
    bin_weights = compute_uniform_weights(bin_counts)
    bin_counts = tf.cast(bin_counts, tf.float32)
    assert bin_weights.shape == bin_counts.shape, f"Expected shape {bin_counts.shape}, got {bin_weights.shape}"
    assert tf.reduce_sum(bin_weights).numpy() == pytest.approx(tf.reduce_sum(
        bin_counts)), f"Expected sum {tf.reduce_sum(bin_counts)}, got {tf.reduce_sum(bin_weights)}"
    dist = bin_weights.numpy() * bin_counts.numpy()
    assert np.all(np.isclose(dist, dist[0])), f"Epected all elements to be equal, got {bin_weights*bin_counts}"
