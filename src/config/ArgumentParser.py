from tap import Tap
from typing import Optional

class ArgumentParser(Tap):
    # basic
    seed: int = 42   # Random seed.
    threads: int = 1   # Maximum number of threads to use.
    debug: bool = False   # Debug mode.
    data_path: str = 'data'   # Path to data folder containing folder of *.root files.
    logdir: str = 'logs'   # Path to log directory.
    tb_update_freq: int = 100   # Frequency of TensorBoard updates.
    
    # data options
    labels: list[str] = ["gluon", "quark"]   # List of labels to use.
    num_labels: int = len(labels)  # Number of labels to use.
    input_size: int = 9   # Number of input features.
    target: str = 'jets_truth_partonPDG'  #'taus_truth_matchJetPdgId' 
    variables: list[str] = ['jets_Jvt', 'jets_Timing', 'jets_chf', 'jets_eta', 'jets_fmax',
                            'jets_m', 'jets_phi', 'jets_pt', 'jets_sumPtTrk']
                        # [ 'taus_seedJetE', 'taus_seedJetEta', 'taus_seedJetPhi',
                        # 'taus_seedJetPt',  'taus_dRminToJet', 'taus_TrackWidthPt500TV', 'taus_seedJetWidth']
    weight: Optional[str] = 'weight_mc[:,0]'
    cut: Optional[str] = None
    
    # tf.data.Dataset options
    batch_size: int = 2048  # Batch size.
    validation_step: int = 200  # Validation every n batches.
    reading_size: int = 10_000_000  # Number of events to load at a time.
    num_workers: int = 6  # Number of workers to use when loading data.
    take: Optional[int] = 20_000  # Length of data to use.
    validation_batches: int = 100  # Size of validation dataset.
    dev_size: float = 0.1   # Size of dev dataset.
    test_size: float = 0.01  # Size of test dataset.
    shuffle_buffer: Optional[int] = None  # Size of shuffler buffer.
    
    model: str = 'basic_fc' # Model to use, options: 'basic_fc', 'transformer'.
    normalize: bool = True # Normalize data.
    normalization_size: int = 10_000 # Size of normalization dataset. 

    # basic_fc_model
    if model == 'basic_fc':
        hidden_layers: list[int] = [2*1024,2*1024]   # Hidden layer sizes.
        dropout: float = 0.5   # Dropout after FC layers.
    
    # transformer_model
    if model == 'transformer':
        warmup_steps: int = 100 # Number of steps to warmup for
        transformer_dropout: float = 0.2
        transformer_expansion: float = 4
        transformer_heads: int = 4
        transformer_layers: int = 4
        embeding: str = 'rnn' # Embedding to use, options: 'cnn', 'rnn'.
        last_fc_size: int = 512 # Size of last fully connected layer.
        
        if embeding == 'rnn':
            embed_dim: int = 128
            rnn_cell: str = 'LSTM'
            rnn_cell_dim: list[int] = []
            
        elif embeding == 'cnn':
            conv_filters: list[int] = [4, 16, 64, 128]
            conv_kernel_size: int = 3
            
    if model == 'BDT':
            num_trees:int = 10
            growing_strategy:str="LOCAL"
            max_depth:int = 3
            split_axis:str="SPARSE_OBLIQUE"
            categorical_algorithm:str="CART"
            
    #training 
    epochs: int = 2 # Number of epochs.
    label_smoothing: float = 0.1 # Label smoothing.
    learning_rate: float = 0.001
    decay_steps: int = 31266*epochs # Number of steps to decay for.
    
    # TODO
    weight_decay: float = 0.0
    
    
    