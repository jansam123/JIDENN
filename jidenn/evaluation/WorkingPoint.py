from __future__ import annotations
from typing import List, Union, Optional
from jidenn.config.eval_config import BinningConfig
import pandas as pd
import numpy as np
import pickle
from jidenn.histogram.BinnedVariable import BinnedVariable


class WorkingPoint(BinnedVariable):

    def __init__(self,
                 binning: BinningConfig,
                 thresholds: Union[List[float], np.ndarray],
                 working_point: Optional[float] = None):

        super().__init__(binning, thresholds)
        self.working_point = working_point

    def set_working_point(self, working_point: float) -> None:
        self.working_point = working_point
        
