Module jidenn.model_builders
============================
Module for building models. It separates the model building process into two parts: model initialization 
and model building with output layer, optimizer, loss, and metrics, with subsequent compilation.
This provdes a two step facade between model definition, model initialization and model building with output layer, compilation.

During the compilation, the optimizer, loss, and metrics are defined. The optimizer is defined by the optimizer config, `jidenn.config.Optimizer`.

Sub-modules
-----------
* jidenn.model_builders.LearningRateSchedulers
* jidenn.model_builders.ModelBuilder
* jidenn.model_builders.callbacks
* jidenn.model_builders.model_initialization
* jidenn.model_builders.multi_gpu_strategies
* jidenn.model_builders.normalization_initialization
* jidenn.model_builders.optimizer_initialization