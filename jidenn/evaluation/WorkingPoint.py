from jidenn.config.eval_config import Binning
import pandas as pd
import numpy as np


class WorkingPoint:

    def __init__(self,
                 binned_variable: str,
                 binning: Binning,
                 working_point: float,):

        self.binned_variable = binned_variable
        self.binning = binning
        self.working_point = working_point
        self.bins = self._create_bins(binning)


    def _create_bins(self, binning: Binning) -> np.ndarray:
        if binning.log_bin_base is not None:
            bins = np.logspace(np.log10(binning.min_bin), np.log10(binning.max_bin),
                               binning.bins, base=binning.log_bin_base)
        else:
            bins = np.linspace(binning.min_bin, binning.max_bin, binning.bins)
        return bins
    
    def _create_intervals(self) -> pd.IntervalIndex:
        return pd.IntervalIndex.from_breaks(self.bins, closed='left')