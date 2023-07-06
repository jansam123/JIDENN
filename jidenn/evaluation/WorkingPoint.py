from jidenn.config.eval_config import Binning
import pandas as pd
import numpy as np

class WorkingPoint:

    def __init__(self,
                 binning: Binning,
                 working_point: float,):

        self.binning = binning
        self.working_point = working_point
        self.bins = self._create_bins(binning)
        self.intervals = self._create_intervals()


    def _create_bins(self, binning: Binning) -> np.ndarray:
        if binning.log_bin_base is not None:
            bins = np.logspace(np.log10(binning.min_bin), np.log10(binning.max_bin),
                               binning.bins + 1, base=binning.log_bin_base)
        else:
            bins = np.linspace(binning.min_bin, binning.max_bin, binning.bins + 1)
        return bins
    
    def _create_intervals(self) -> pd.IntervalIndex:
        return pd.IntervalIndex.from_breaks(self.bins, closed='right')

    def set_thresholds(self, thresholds: Union[List[float], np.ndarray]):
        if len(thresholds) != self.binning.bins:
            raise ValueError(f'Length of thresholds {len(thresholds)} does not match number of bins {self.binning.bins}')
        if isinstance(thresholds, list):
            thresholds = np.array(thresholds)
        self.thresholds = thresholds

    def __str__(self):
        dict_values = {'binning': self.intervals}
        if hasattr(self, 'thresholds'):
            dict_values['thresholds'] = self.thresholds
        else:
            dict_values['thresholds'] = np.full(self.binning.bins, np.nan)
                       
        df = pd.DataFrame(dict_values)
        return df.to_string(index=False)
        
        
    
# binning = Binning('pt,', 10, 100, 0, None)
# wp = WorkingPoint(binning, 0.5)
# wp.set_thresholds([0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1.0])
# print(wp)