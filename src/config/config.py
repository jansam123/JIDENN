from dataclasses import dataclass

from .config_subclasses import Params, Data, Dataset, Preprocess, Optimizer, Models

    
@dataclass
class JIDENNConfig:    
    params: Params
    data: Data
    dataset: Dataset
    preprocess: Preprocess
    optimizer: Optimizer
    models: Models


    
