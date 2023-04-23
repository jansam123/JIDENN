"""
Module for building models. It separates the model building process into two parts: model initialization 
and model building with output layer, optimizer, loss, and metrics, with subsequent compilation.
This provdes a two step facade between model definition, model initialization and model building with output layer, compilation.

During the compilation, the optimizer, loss, and metrics are defined. The optimizer is defined by the optimizer config, `jidenn.config.Optimizer`. 
"""