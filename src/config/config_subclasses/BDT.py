from dataclasses import dataclass

@dataclass
class BDT:
    num_trees: int
    growing_strategy: str
    max_depth: int
    split_axis: str
    categorical_algorithm: str
    