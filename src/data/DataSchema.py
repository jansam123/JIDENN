from src.config.ArgumentParser import ArgumentParser
from dataclasses import dataclass, field
from typing import Optional, List, Callable

@dataclass
class DataSchema:
    variables: List[str] = field(default_factory=lambda: ArgumentParser().variables)
    target:str = ArgumentParser().target
    weight: Optional[str] = ArgumentParser().weight
    num_labels: int = ArgumentParser().num_labels
    gluon: int = 0
    quark: int = 1
    raw_qluon: int = 21
    raw_quarks: List[int] = field(default_factory=lambda: [1,2,3,4,5])
    raw_unknown: List[int] = field(default_factory=lambda: [-1, -999])
    
    
    @property
    def label_mapping(self) -> Callable[[int], int]:
        def mapping(x):
            if x == self.raw_qluon:
                return self.gluon
            else:
                return self.quark
        return mapping