from __future__ import annotations
from typing import List, Union, Optional
from jidenn.config.eval_config import BinningConfig
import pandas as pd
import numpy as np
import pickle


class Binning:
    """Binning of a variable. Simple wrapper around `np.linspace` to create bins. Can be constructed from a `BinningConfig` using `from_config` method.
    
    Args:
        variable (str): Name of the variable.
        bins (Union[int, List[int]]): Number of bins or list of bin edges.
        max_bin (Union[float, int]): Maximum bin edge.
        min_bin (Union[float, int]): Minimum bin edge.
        log_bin_base (Optional[int]): Base for logarithmic binning. If `None`, linear binning is used.
    
    """
    def __init__(self, variable: str, bins: Union[int, List[int]], min_bin: Union[float, int], max_bin: Union[float, int], log_bin_base: Optional[int]=None):
        self.variable = variable
        self._max_bin = max_bin
        self._min_bin = min_bin
        self._log_bin_base = log_bin_base
        self._bins = self._create_bins(bins, max_bin, min_bin, log_bin_base)
        
    @staticmethod
    def from_config(config: BinningConfig) -> Binning:
        """Create a `Binning` from a `BinningConfig`.
        Args:
            config (BinningConfig): Configuration for binning a continuous variable.
            
        Returns:
            Binning: Binning of the variable.
        """
        
        return Binning(config.variable, config.bins, config.max_bin, config.min_bin, config.log_bin_base)

    def _create_bins(self, bins: Union[int, List[int]], max_bin: Union[float, int], min_bin: Union[float, int], log_bin_base: Optional[int]) -> np.ndarray:
        """Helper method to create bins.  
        
        Args:
            bins (Union[int, List[int]]): Number of bins or list of bin edges.
            max_bin (Union[float, int]): Maximum bin edge.
            min_bin (Union[float, int]): Minimum bin edge.
            log_bin_base (Optional[int]): Base for logarithmic binning. If `None`, linear binning is used.
            
        Returns:
            np.ndarray: Array of bin edges. 
        """
        if log_bin_base is not None and log_bin_base != 0 and min_bin is not None and max_bin is not None and isinstance(bins, int):
            min_val = np.log(self.min_bin) / \
                np.log(log_bin_base) if log_bin_base != 0 else np.log(
                    min_bin)
            max_val = np.log(self.max_bin) / \
                np.log(log_bin_base) if log_bin_base != 0 else np.log(
                    max_bin)
            return np.logspace(min_val, max_val,
                               bins + 1, base=log_bin_base if log_bin_base != 0 else np.e)
        elif min_bin is not None and max_bin is not None and isinstance(bins, int):
            return np.linspace(
                min_bin, max_bin, bins + 1)
        else:
            return np.array(bins)

    @property
    def bins(self) -> np.ndarray:
        """Array of bin edges."""
        return self._bins

    @property
    def n_bins(self) -> int:
        """Number of bins."""
        return len(self.bins) - 1

    @property
    def max_bin(self) -> Union[float, int]:
        """Maximum bin edge."""
        return self._max_bin

    @property
    def min_bin(self) -> Union[float, int]:
        """Minimum bin edge."""
        return self._min_bin

    @property
    def intervals(self) -> pd.IntervalIndex:
        """`pd.IntervalIndex` of the bins."""
        return pd.IntervalIndex.from_breaks(self.bins)
    
    @property
    def string_intervals(self) -> List[str]:
        """List of string representations of the bins."""
        return list(self.intervals.astype(str).values)
    
    @property
    def bin_mids(self) -> np.ndarray:
        """Array of bin midpoints."""
        return self.intervals.mid.values
    
    @property
    def bin_widths(self) -> np.ndarray:
        """Array of bin widths."""
        return self.intervals.length.values
    
    @property
    def bin_edges(self) -> np.ndarray:
        """Array of bin edges."""
        return self.bins
    
    def __str__(self) -> str:
        return f'Binning({self.variable}, {self.bins})'
    def __repr__(self) -> str:
        return f'Binning({self.variable}, {self.bins})'
    
    def __eq__(self, other: Binning) -> bool:
        return (self.variable == other.variable) and np.allclose(a=self.bins, b=other.bins)
    
    def __ne__(self, other: Binning) -> bool:
        return not self.__eq__(other)

class BinnedVariable:

    def __init__(self,
                 binning: BinningConfig,
                 values: Union[List[float], np.ndarray]):

        self.binning = Binning.from_config(binning)
        # self.bins = binning.bins
        self.intervals = self._create_intervals()
        self.set_values(values)

    def _create_intervals(self) -> pd.IntervalIndex:
        return pd.IntervalIndex.from_breaks(self.binning.bins)

    def set_values(self, values: Union[List[float], np.ndarray]) -> None:
        if isinstance(values, list):
            values = np.array(values)
        self._values = values

    @property
    def values(self) -> np.ndarray:
        return self._values

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> BinnedVariable:
        with open(path, 'rb') as f:
            return pickle.load(f)

    def __getitem__(self, key: pd.Interval) -> float:
        if isinstance(key, pd.Interval):
            return self._values[self.binning.intervals.get_loc(key)]
        elif isinstance(key, int):
            return self._values[key]
        elif isinstance(key, str):
            return self._values[self.binning.string_intervals.get_loc(key)]
        else:
            raise KeyError(
                f'Key {key} not understood. Must be pd.Interval or int')

    @property
    def bins(self) -> np.ndarray:
        return self.binning.bins
    
    def __str__(self) -> str:
        return f'BinnedVariable({self.binning.variable}, {self.binning.bins})'