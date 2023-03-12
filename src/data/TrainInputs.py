import tensorflow as tf
import numpy as np
from typing import Union, Literal, Callable, Dict, Tuple, List, Optional
#
from .utils.transformations import to_e_px_py_pz


PFOVariables = Dict[str, tf.RaggedTensor]
InteractionVariables = Dict[str, tf.RaggedTensor]
InteractingPFOVariables = Tuple[PFOVariables, PFOVariables]
JetVariables = Dict[str, tf.Tensor]

ROOTVariables = Dict[str, Union[tf.Tensor, tf.RaggedTensor]]
JIDENNVariables = Dict[Literal['perJet', 'perJetTuple', 'perEvent'], ROOTVariables]
JetVariableMapper = Callable[[JIDENNVariables], JetVariables]


