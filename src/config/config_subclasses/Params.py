from dataclasses import dataclass

@dataclass
class Params:
    model: str # Model to use, options: 'basic_fc', 'transformer'.
    epochs: int # Number of epochs.
    label_smoothing: float # Label smoothing.
    learning_rate: float
    seed: int   # Random seed.
    threads: int   # Maximum number of threads to use.
    debug: bool   # Debug mode.
    data_path: str   # Path to data folder containing folder of *.root files.
    logdir: str   # Path to log directory.
    #TODO
    decay_steps: int # Number of steps to decay for.
    weight_decay: float