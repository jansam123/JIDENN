from dataclasses import dataclass
from typing import Callable

@dataclass
class Data:
    labels: list[str]    # list of labels to use.
    num_labels: int   # Number of labels to use.
    input_size: int    # Number of input features.
    target: str  
    variables: list[str] 
    weight: str | None 
    cut: str | None 
    tttree_name: str  # Name of the tree in the root file.
    gluon: int 
    quark: int 
    raw_gluon: int 
    raw_quarks: list[int] 
    raw_unknown: list[int]
    
    
