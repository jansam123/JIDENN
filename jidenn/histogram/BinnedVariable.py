from __future__ import annotations
from dataclasses import dataclass
from typing import List, Union, Optional
from jidenn.config.eval_config import BinningConfig     
import pandas as pd
import numpy as np
import pickle



class Binning:
    def __init__(self, binning_cfg: BinningConfig):
        self.binning_cfg = binning_cfg
        self._bins = self._create_bins()

    def _create_bins(self) -> np.ndarray:
        if self.log_bin_base is not None:
            min_val = np.log(self.min_bin) / \
                np.log(self.log_bin_base) if self.log_bin_base != 0 else np.log(
                    self.min_bin)
            max_val = np.log(self.max_bin) / \
                np.log(self.log_bin_base) if self.log_bin_base != 0 else np.log(
                    self.max_bin)
            bins = np.logspace(min_val, max_val,
                               self.bins + 1, base=self.log_bin_base if self.log_bin_base != 0 else np.e)
        elif self.min_bin is not None and self.max_bin is not None:
            bins = np.linspace(
                self.min_bin, self.max_bin, self.bins + 1)
        else:
            bins = np.array(self.bins)
        return bins

    @property
    def bins(self) -> np.ndarray:
        return self._bins

    @property
    def n_bins(self) -> int:
        return len(self.bins) - 1


class BinnedVariable:

    def __init__(self,
                 binning: Binning,
                 values: Union[List[float], np.ndarray]):

        self.binning = binning
        self.bins = binning._create_bins()
        self.intervals = self._create_intervals()
        self.set_values(values)

    def _create_intervals(self) -> pd.IntervalIndex:
        return pd.IntervalIndex.from_breaks(self.bins)

    def set_values(self, values: Union[List[float], np.ndarray]) -> None:
        # if len(values) != self.binning.bins:
        #     raise ValueError(f'Length of thresholds {len(values)} does not match number of bins {self.binning.bins}')
        if isinstance(values, list):
            values = np.array(values)
        self._values = values

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def string_intervals(self) -> List[str]:
        return list(self.intervals.astype(str).values)

    def __str__(self) -> str:
        dict_values = {'binning': self.intervals}
        if hasattr(self, 'thresholds'):
            dict_values['thresholds'] = self._values
        else:
            dict_values['thresholds'] = np.full(self.binning.bins, np.nan)

        df = pd.DataFrame(dict_values)
        return df.to_string(index=False)

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> BinnedVariable:
        with open(path, 'rb') as f:
            return pickle.load(f)

    def __getitem__(self, key: pd.Interval) -> float:
        if isinstance(key, pd.Interval):
            return self._values[self.intervals.get_loc(key)]
        elif isinstance(key, int):
            return self._values[key]
        elif isinstance(key, str):
            return self._values[self.string_intervals.get_loc(key)]
        else:
            raise KeyError(
                f'Key {key} not understood. Must be pd.Interval or int')

    def get_bin_mids(self, scale: float) -> np.ndarray:
        return self.intervals.mid.values * scale

    def get_bin_widths(self, scale: float) -> np.ndarray:
        return self.intervals.length.values * scale
