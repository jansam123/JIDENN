from src.config.config import ArgumentParser
from dataclasses import dataclass, field
from typing import Callable

@dataclass
class DataSchema:
    variables: list[str] = field(default_factory=lambda: ArgumentParser().variables)
    target:str = ArgumentParser().target
    weight:str | None = ArgumentParser().weight
    num_labels: int = ArgumentParser().num_labels
    
    gluon: int = 0
    quark: int = 1
    raw_qluon: int = 21
    raw_quarks: list[int] = field(default_factory=lambda: [1,2,3,4,5])
    raw_unknown: list[int] = field(default_factory=lambda: [-1, -999])
    
    
    @property
    def label_mapping(self) -> Callable[[int], int]:
        def mapping(x):
            if x == self.raw_qluon:
                return self.gluon
            else:
                return self.quark
        return mapping