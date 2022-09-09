from dataclasses import dataclass

@dataclass
class Preprocess:
    normalize: bool  # Normalize data.
    normalization_size: int | None  # Size of normalization dataset. 