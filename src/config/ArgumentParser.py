from tap import Tap


class ArgumentParser(Tap):
    # basic
    seed: int = 42   # Random seed.
    threads: int = 1   # Maximum number of threads to use.
    debug: bool = False   # Debug mode.
    data_path: str = 'data'   # Path to data folder containing folder of *.root files.
    logdir: str = 'logs'   # Path to log directory.
    tb_update_freq: int = 100   # Frequency of TensorBoard updates.
    
    # data options
    labels: list[str] = ["gluon", "quark"]   # list of labels to use.
    num_labels: int = len(labels)  # Number of labels to use.
    input_size: int = 9   # Number of input features.
    target: str = 'jets_truth_partonPDG'  #'taus_truth_matchJetPdgId' 
    variables: list[str] = ['jets_Jvt', 'jets_Timing', 'jets_chf', 'jets_eta', 'jets_fmax',
                            'jets_m', 'jets_phi', 'jets_pt', 'jets_sumPtTrk']
                        # [ 'taus_seedJetE', 'taus_seedJetEta', 'taus_seedJetPhi',
                        # 'taus_seedJetPt',  'taus_dRminToJet', 'taus_TrackWidthPt500TV', 'taus_seedJetWidth']
    weight: str | None = 'weight_mc[:,0]'
    cut: str | None = None
    
    # tf.data.Dataset options
    batch_size: int = 2048  # Batch size.
    validation_step: int = 200  # Validation every n batches.
    reading_size: int = 1_000  # Number of events to load at a time.
    num_workers: int = 6  # Number of workers to use when loading data.
    take: int | None = 8_000  # Length of data to use.
    validation_batches: int = 100  # Size of validation dataset.
    dev_size: float = 0.1   # Size of dev dataset.
    test_size: float = 0.01  # Size of test dataset.
    shuffle_buffer: int | None = None  # Size of shuffler buffer.
    
    #preprocess options
    normalize: bool = True # Normalize data.
    normalization_size: int = 100 # Size of normalization dataset. 
    
    model: str = 'BDT' # Model to use, options: 'basic_fc', 'transformer'.

    # basic_fc_model
    if model == 'basic_fc':
        hidden_layers: list[int] = [2*1024,2*1024]   # Hidden layer sizes.
        dropout: float = 0.5   # Dropout after FC layers.
    
    # transformer_model
    if model == 'transformer':
        warmup_steps: int = 100 # Number of steps to warmup for
        transformer_dropout: float = 0.2
        transformer_expansion: float = 4 #4,  number of hidden units in FFN is transformer_expansion * embed_dim
        transformer_heads: int = 8 #12, must divide embed_dim
        transformer_layers: int = 6 #6,12
        last_fc_size: int = 512 # Size of last fully connected layer.
        embed_dim: int = 256  #512
        
            
    if model == 'BDT':
            num_trees:int = 10
            growing_strategy:str="LOCAL"
            max_depth:int = 3
            split_axis:str="SPARSE_OBLIQUE"
            categorical_algorithm:str="CART"
            
    #training 
    epochs: int = 1 # Number of epochs.
    label_smoothing: float = 0.1 # Label smoothing.
    learning_rate: float = 0.001
    decay_steps: int = 31266*epochs # Number of steps to decay for.
    
    # TODO
    weight_decay: float = 0.0
    
    
    