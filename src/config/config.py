from dataclasses import dataclass

from .config_subclasses import BasicFC, Highway, BDT, Params, Data, Dataset, Preprocess, Transformer, DeParT, ParT

    
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
    part: ParT
    depart: DeParT

    
