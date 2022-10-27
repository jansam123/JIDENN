from dataclasses import dataclass
from .config_subclasses import *

    
@dataclass
class JIDENNConfig:    
    params: Params
    data: Data
    dataset: Dataset
    preprocess: Preprocess
    basic_fc: BasicFC
    transformer: Transformer
    bdt: BDT
    highway: Highway

    
