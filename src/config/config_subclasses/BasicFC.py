
from dataclasses import dataclass

@dataclass
class BasicFC:
    hidden_layers: list[int]    # Hidden layer sizes.
    dropout: float    # Dropout after FC layers.