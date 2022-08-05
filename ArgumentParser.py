from tap import Tap
from typing import Optional

class ArgumentParser(Tap):
    # basic
    seed: int = 42   # Random seed.
    threads: int = 1   # Maximum number of threads to use.
    debug: bool = False   # Debug mode.
    data_path: str = 'data'   # Path to data folder containing folder of *.root files.
    logdir: str = 'logs'   # Path to log directory.
    tb_update_freq: str = '10'   # Frequency of TensorBoard updates.
    
    # dataset
    batch_size: int = 1024  # Batch size.
    validation_step: int = 200  # Validation every n batches.
    reading_size: int = 1_000  # Number of events to load at a time.
    num_workers: int = 6  # Number of workers to use when loading data.
    take: Optional[int] = None  # Length of data to use.
    validation_batches: int = 100  # Size of validation dataset.
    dev_size: float = 0.1   # Size of dev dataset.
    test_size: float = 0.01  # Size of test dataset.
    
    model: str = 'transformer' # Model to use, options: 'basic_fc', 'transformer'.

    # basic_fc_model
    if model == 'basic_fc':
        hidden_layers: list[int] = [2*1024,2*1024, 512]   # Hidden layer sizes.
        dropout: float = 0.5   # Dropout after FC layers.
    
    # transformer_model
    if model == 'transformer':
        transformer_dropout: float = 0.
        transformer_expansion: float = 2
        transformer_heads: int = 32
        transformer_layers: int = 16
        embeding: str = 'rnn' # Embedding to use, options: 'cnn', 'rnn'.
        
        if embeding == 'rnn':
            embed_dim: int = 256
            rnn_cell: str = 'LSTM'
            rnn_cell_dim: list[int] = [256, 256]
            
        elif embeding == 'cnn':
            conv_filters: list[int] = [4, 16, 64, 128]
            conv_kernel_size: int = 3
        
        
            
    #training 
    epochs: int = 1 # Number of epochs.
    label_smoothing: float = 0.0
    warmup_steps: int = 100 # Number of steps to warmup for.
    
    # TODO
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    
    
    