"""
This package contains the code for the JIDENN project.
It contains all the necessary code to train and evaluate the models used for jet identification.

The process starts with the `jidenn.data` package, which contains the code to load the data,
preprocess it, and create the datasets used for training and evaluation. It is recommended to
convert the ROOT file to saved `tf.data.Dataset` objects, as this is much faster than loading directly.
On top of that it provides an easy way to create own training inputs, with the 
`jidenn.data.TrainInput` module.

The models are defined in the `jidenn.models` package. There are both, models that take the 
jet constituents as input, and models that take the high-level jet variables already present in the
ROOT file as input. 

The model creation is done with the `jidenn.model_builders` module. This module contains 
multiple facades for the model creation, separating individual steps to make it easier to
extend the code at any point. 

All configuration is done with the `hydra` package. The configuration files are located in the
`jidenn/config` folder. The `jidenn.config` module contains the dataclasses that connect the 
`*.yaml` files to the code.

Evaluation is done with the `jidenn.evaluation` module. This module contains the code to evaluate
custom metrics, and to plot the results. 

"""