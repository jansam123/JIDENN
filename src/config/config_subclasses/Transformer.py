from dataclasses import dataclass

@dataclass
class Transformer:
    warmup_steps: int # Number of steps to warmup for
    transformer_dropout: float # Dropout after FFN layer.
    transformer_expansion: int  #4,  number of hidden units in FFN is transformer_expansion * embed_dim
    transformer_heads: int  #12, must divide embed_dim
    transformer_layers: int  #6,12
    last_hidden_layer: int  # Size of last fully connected layer.
    embed_dim: int 
    