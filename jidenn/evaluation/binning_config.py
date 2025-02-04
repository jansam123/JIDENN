from __future__ import annotations
from dataclasses import dataclass
from typing import Union

class InvalidBinningConfig(Exception):
    def __init__(self, config: BinningConfig, *args):
        first_args = 'Invalid config' + str(config) + '. ' + args[0]
        return super().__init__(first_args, *args[1:])


@dataclass 
class BinningConfig():
    variable: str
    bins: list[Union[float, int]]
    name: Union[str, list[Union[str, None]], None] = None
    cut: Union[str, list[Union[str, None]], None] = None

    def __post_init__(self):
        
        if isinstance(self.name, str):
            if not isinstance(self.cut, str):
                raise InvalidBinningConfig(self, '`name` and `cut` must be the same type')
                
        elif isinstance(self.name, list):
            if not isinstance(self.cut, list):
                raise InvalidBinningConfig(self, '`name` and `cut` must be the same type.')
            if len(self.name) != len(self.cut):
                raise InvalidBinningConfig(self, 'Length of `cut` and `name` must be the same.')
            
        elif self.name is None and not self.cut is None:
            raise InvalidBinningConfig(self, '`name` and `cut` must be the same type.')
            
            
        
            
            
            
        

BINNINGS = [
    BinningConfig(
        variable='jets_pt',
        bins=[20_000, 60_000, 160_000, 400_000, 800_000, 1_300_000, 1_800_000, 2_500_000],
    ),
    # BinningConfig(
    #     variable='jets_eta',
    #     bins=[0., 0.9, 1.8, 2.7, 3.6, 4.5],
    # ),
    # BinningConfig(
    #     variable='jets_eta',
    #     bins=[0., 0.9, 1.8, 2.7, 3.6 ,4.5],
    #     cut=['jets_pt>20_000 and jets_pt<60_000', 'jets_pt>60_000 and jets_pt<160_000', 'jets_pt>160_000 and jets_pt<500_000', 'jets_pt>500_000',],
    #     name=['20-60', '60-160', '160-500', '500+']
    # ),
    # BinningConfig(
    #     variable='jets_Constituent_n',
    #     bins=[0, 10, 20, 40, 60, 80],
    # ),
    # BinningConfig(
    #     variable='jets_TopoTower_n',
    #     bins=[0, 10, 20, 40, 60, 80],
    # ),
    # BinningConfig(
    #     variable='jets_Constituent_n+jets_TopoTower_n',
    #     bins=[0, 10, 20, 40, 60, 80, 100],
    # ),
    # BinningConfig(
    #     variable='jets_index',
    #     bins=[0, 1, 2, 3, 4, 5, 10],
    #     cut=[None, 'jets_pt>20_000 and jets_pt<60_000', 'jets_pt>60_000 and jets_pt<160_000', 'jets_pt>160_000 and jets_pt<500_000', 'jets_pt>500_000',],
    #     name=[None, '20-60', '60-160', '160-500', '500+']
    # ),
]