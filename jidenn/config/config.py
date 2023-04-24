"""
General configurations for JIDENN. It includes training and data preparation configurations.
The model configurations are defined separately in `jidenn.config.model_config`, but are
in the same `.yaml` file as the general configurations.

The data is exspected to have the following structure:

- the main folder contains the folders `train`, `dev` and `test` which contain the saved `tf.data.Dataset`s.
- optionally, the main folder can contain subfolders which then each contain the folders `train`, `dev` and `test`.
```bash
main_folder/
    train/
    dev/
    test/
# or
main_folder/
    subfolder1/
        train/
        dev/
        test/
    subfolder2/
        train/
        dev/
        test/
    ...
```
The saving of the `tf.data.Dataset`s is done as:
```python
tf.data.Dataset.save('main_folder/train')
...
# or
tf.data.Dataset.save('main_folder/subfolder1/train')
...
```

The saved datasets are expected to be flattened to the jet level.
"""
from dataclasses import dataclass
from typing import List, Optional, Literal

from jidenn.config.model_config import FC, Highway, BDT, Transformer, DeParT, ParT, PFN


@dataclass
class Variables:
    """Variable names loaded from the saved `tf.data.Dataset`s. The `element_spec` of the loaded dataset
    is expected to be `Dict[str, Union[tf.Tensor, tf.RaggedTensor]]` where the keys are the variable names.
    From these variables, only variables defined in `per_jet` and `per_jet_tuple` and `per_event` are used.
    They are then clustered into a dictionary of the following structure 
    `Dict[Literal['perEvent','perJet','perJetTuple'], Dict[str, Union[tf.Tensor, tf.RaggedTensor]]]`.
    Example:
    ```python
    # element_spec of the loaded dataset, ROOTDataset
    element_spec_before = {
        'jets_PFO_m': RaggedTensorSpec(TensorShape([None]), tf.float32, 0, tf.int64),
        'jets_pt': TensorSpec(shape=(), dtype=tf.float32, name=None)
        'HLT_j60': TensorSpec(shape=(), dtype=tf.bool, name=None),
    }
    # element_spec after processing with JIDENNDataset
    element_spec_after = {
        'perEvent': {'HLT_j60': TensorSpec(shape=(), dtype=tf.bool, name=None)},
        'perJet': {'jets_pt': TensorSpec(shape=(), dtype=tf.float32, name=None)}
        'perJetTuple': {'jets_PFO_m': RaggedTensorSpec(TensorShape([None]), tf.float32, 0, tf.int64)},
    }
    ```
    Optionally, the string may be expression which combine multiple variables, or slice them.
    See the `jidenn.data.string_conversions.Expression` documentation for more information. 

    These variables are then used to construct the model inputs with the `jidenn.data.TrainInput` class.

    Args:
        per_jet (List[str]): Variables to be loaded from the dataset and clustered into `perJet`.
            Each variable is expected to have single input per jet, e.g. `jets_pt` 
            which is a single value for the jet transverse momentum.
        per_jet_tuple (List[str]): Variables to be loaded from the dataset and clustered into `perJetTuple`.
            Each variable is expecetd to have multiple inputs per jet, e.g. `jets_PFO_m` which is a list of
            masses of the constituents of the jet.
        per_event (List[str]): Variables to be loaded from the dataset and clustered into `perEvent`.
            Each variable is expected to have only one input per event, which is the same for all jets in the event.
            It is tiled to the number of jets in the event.
    """
    per_jet: List[str]
    per_jet_tuple: List[str]
    per_event: List[str]


@dataclass
class Data:
    """ Data configuration for loading the data, constructing a `tf.data.Dataset` and
    labeling the data.

    Args:
        path (str): Path to data folder containing folders of saved `tf.data.Dataset`s. The path **must** contain
            the folders `train`, `dev` and `test` which contain the saved `tf.data.Dataset`s.

        target (str): Name of the target variable inside the saved `tf.data.Dataset`s.

        target_labels (List[List[int]]): Original labels which are going to be changed by clustering into lists. 
            The following example will cluster all jets with `target in [21]` into one label,
            and all jets with `target in [1, 2, 3, 4, 5, 6]` into the second label:

                target_labels:
                    - [21]
                    - [1, 2, 3, 4, 5, 6]

        labels (List[str]): List of names of the labels defined by the order in `target_labels`.

                labels:
                    - gluon
                    - quark

        variable_unknown_labels (List[int]): List of unknown labels corresponding to undefined `target` (e.q. `[-1, -999]`). 
            These will be omiited from the dataset.

        variables (Variables): Variables to loaded from the dataset separated into `per_jet` and `per_event` and `per_jet_tuple` 
            using the `Variables` dataclass. The actual variables used as a training input are defined 
            from these in `jidenn.data.TrainInput`.

        weight (Optional[str]): Name of the weight variable inside the saved `tf.data.Dataset`s. If `None`, 
            no weights are used otherwise the weights are passed as a third input to the model.

        cut (Optional[str]): Cut to apply to the dataset. If `None`, no cut is applied. String are parsed using the `jidenn.data.Cut` class.
            See the `jidenn.data.Cut` documentation for more information of the cut syntax.

        subfolders (Optional[List[str]]): Folders inside the `path` to use. If `None`, the `path` is used directly. Each subfolder
            must contain the folders `train`, `dev` and `test` which contain the saved `tf.data.Dataset`s.
            These subfolders are used to combine multiple JZ slices into one dataset.

        subfolder_cut (Optional[List[str]]): Cuts to apply to the individual subfolders separately. If `None`, no cut is applied. 
            Must be the same length as `subfolders`. String are parsed using the `jidenn.data.Cut` class.
            See the `jidenn.data.Cut` documentation for more information of the cut syntax.

        subfolder_weights (Optional[List[float]]): Weights to apply to the individual subfolders separately when combining. 
            `None` is viable only if `subfolders` is `None`. 

        cached (Optional[str]): **Untested.** Path to cached data. If `None`, no cached data is used. If `cached` is not `None`, the `path` is ignored.
    """
    path: str   # Path to data folder containing folder of *.root files.
    target: str
    target_labels: List[List[int]]   # Original labels.
    labels: List[str]    # list of labels to use.
    variable_unknown_labels: List[int]
    variables: Variables
    weight: Optional[str]
    cut: Optional[str]
    subfolders: Optional[List[str]]   # Slices of JZ to use.
    subfolder_cut: Optional[List[str]]   # Cut to apply to JZ slices.
    subfolder_weights: Optional[List[float]]  # Weights to apply to JZ slices.
    cached: Optional[str]   # Path to cached data.


@dataclass
class Dataset:
    """ Dataset configuration for preparing the `tf.data.Dataset` for training.
    Args:
        epochs (int): Number of epochs.
        batch_size (int): Batch size.
        take (Optional[int]): Length of data to use. i.e. number of jets. 
            If `None`, the whole `train` dataset is used.
        dev_size (float): Size of dev dataset as a fraction of the `take` length.
            If `take` is `None`, the size is omitted and whole `dev` dataset is used.
        test_size (float): Size of test dataset as a fraction of the `take` length.
            If `take` is `None`, the size is omitted and whole `test` dataset is used.
        shuffle_buffer (Optional[int]): Size of shuffler buffer, if `None`, no shuffling is used.
            `shullfe_buffer` samples are shuffled before each epoch.
    """
    epochs: int  # Number of epochs.
    batch_size: int   # Batch size.
    take: Optional[int]   # Length of data to use.
    dev_size: float   # Size of dev dataset.
    test_size: float  # Size of test dataset.
    shuffle_buffer: Optional[int]   # Size of shuffler buffer.


@dataclass
class General:
    """Basic configuration for the training.

    Args:
        model (str): Model to use, options: `fc`, `highway`, `pfn`, `efn`, `transformer`, `part`, `depart`, `bdt`.
        seed (int): Random seed. Used for reproducibility.
        threads (int, optional): Maximum number of threads to use. `None` or 0 uses all threads.
        debug (bool): Debug mode. If `True`, tensorflow uses the `Eager` mode.
        base_logdir (str): Path to log directory where subfolders are created for each training session.
        logdir (str): Path to log directory of a given training session. Could be set manually,
            but using `${general.base_logdir}/${now:%Y-%m-%d}__${now:%H-%M-%S}` is recommended, 
            as it creates a unique folder for each training session inside the `base_logdir`.
        checkpoint (str, optional): Path to a checkpoint inside `logdir` checkpoint. If `None`, no checkpoint is made.
        backup (str, optional): Path to a backup of the model inside `logdir` checkpoint. If `None`, no backup is made.
        load_checkpoint_path (str, optional): Path to a checkpoint to load. If `None`, no checkpoint is loaded.
    """
    model: Literal['fc', 'highway', 'pfn',
                   'efn', 'transformer', 'part', 'depart', 'bdt']   # Model to use.
    base_logdir: str   # Path to log directory.
    seed: int   # Random seed.
    threads: Optional[int]   # Maximum number of threads to use.
    debug: bool   # Debug mode.
    logdir: str   # Path to log directory.
    checkpoint: Optional[str]   # Make checkpoint.
    backup: Optional[str]   # Backup model.
    load_checkpoint_path: Optional[str]   # Path to checkpoint to load.


@dataclass
class Preprocess:
    """Preprocessing configuration for the `tf.data.Dataset`.

    Args:
        draw_distribution (int, optional): Number of samples to draw distribution for.
            This is useful for interpreting the model based on physical quantities.
            If `None`, no distribution is drawn.    
        normalization_size (int, optional): Number of batches to calculate the mean and standard deviation 
            for each variable used for normalization. If `None`, no normalization is done.

    """
    draw_distribution: Optional[int]   # Number of events to draw distribution for.
    normalization_size: Optional[int]  # Size of normalization dataset.


@dataclass
class Optimizer:
    """Settings for the optimizer.

    Args:
        name (str): Name of the optimizer to use, options: `LAMB`, `Adam`.
        learning_rate (float): Learning rate.
        label_smoothing (float, optional): Label smoothing.
        decay_steps (int, optional): Number of steps to decay the learning rate with cosine decay. 
            If `None`, decay is calculated automatically as `decay_steps = epochs * take / batch_size - warmup_steps`.
        warmup_steps (int, optional): Number of steps to warmup the learning rate with linear warmup.
        beta_1 (float, optional): Beta 1 for Adam and LAMB, default is 0.9.
        beta_2 (float, optional): Beta 2 for Adam and LAMB, default is 0.999.
        epsilon (float, optional): Epsilon for Adam and LAMB, default is 1e-6.
        clipnorm (float, optional): Clipnorm for Adam and LAMB. If `None`, no clipping is done. Default is `None`.
        weight_decay (float, optional): Weight decay for LAMB and Adam, default is 0.0.
    """
    name: Literal['LAMB', 'Adam']
    learning_rate: float
    label_smoothing: Optional[float]
    decay_steps: Optional[int]
    warmup_steps: Optional[int]
    beta_1: Optional[float]
    beta_2: Optional[float]
    epsilon: Optional[float]
    clipnorm: Optional[float]
    weight_decay: Optional[float]


@dataclass
class Models:
    """Configuration for the models. See the documentation for each model for more information."""
    fc: FC
    transformer: Transformer
    bdt: BDT
    highway: Highway
    part: ParT
    depart: DeParT
    pfn: PFN


@dataclass
class JIDENNConfig:
    """A dataclass containing all of the configuration information for a JIDENN training session."""
    general: General
    data: Data
    dataset: Dataset
    preprocess: Preprocess
    optimizer: Optimizer
    models: Models
