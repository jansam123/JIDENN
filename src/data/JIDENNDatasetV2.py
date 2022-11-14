from __future__ import annotations
import tensorflow as tf
from dataclasses import dataclass
#
from utils.CutV2 import Cut
import src.config.config_subclasses as cfg
from src.data.ROOTDataset import ROOTDataset

# import pandas as pd
# import time
# import os@dataclass


ROOTVariables = dict[str, tf.RaggedTensor]


@dataclass
class JIDENNDataset:
    files: list[str]
    variables: cfg.Variables
    element_spec_file: str | None = None
    target: str | None = None
    weight: str | None = None
    event_cut: str | None = None
    jet_cut: str | None = None

    def __post_init__(self):
        @tf.function
        def load_single_ds(file: str) -> tf.data.Dataset:
            root_dataset = ROOTDataset.load(file, self.element_spec_file)
            root_dataset = root_dataset.filter(Cut(self.event_cut)) if self.event_cut is not None else root_dataset
            return root_dataset.dataset

        self._dataset = tf.data.Dataset.from_tensor_slices(self.files).interleave(
            load_single_ds, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
