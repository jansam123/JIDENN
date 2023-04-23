Module jidenn.models
====================
This module contains all the models used for the experiments on jet identification.
They can be basicaly clustered into three categories:

- constituent based models: these models take as input the constituents of the jet 
    (`jidenn.models.Transformer`, `jidenn.models.PFN`, `jidenn.models.EFN`, `jidenn.models.DeParT`, `jidenn.models.ParT`)
- jet variables based models: these models take high-level precomputed jet variables as input
    (`jidenn.models.Highway`, `jidenn.models.FC`)
- jet variables computed from constituents: these models take as input high-level jet variables computed from the constituents
    (`jidenn.models.BDT`)

All models (except `jidenn.models.BDT`) are implemented as a subclass of the `tf.keras.Model` class with
multiple layers as a `tf.keras.layers.Layer` subclass. 
They also have theree common initialization arguments:

- 'input_size': the size of the input
- 'output_layer': the output layer of the model, to allow binary or multi-class classification
- 'preprocess': a preprocessing layer to be applied to the input, eg. a `tf.keras.layers.Normalization` layer (see `jidenn.model_builders.get_normalization`)

The BDT model (`jidenn.models.BDT`) is an implemenatation from `tensorflow_decision_forests`, namely `tfdf.keras.GradientBoostedTreesModel`.
You need to isntall `tensorflow_decision_forests` to use this model.

The specific inputs to the models are defined in the `jidenn.data.TrainInput` module.
Configurations for the models are defined in the `jidenn.config.model_config` module.

The model that we developed is the **Dynamicaly Ehnanced Particle Transformer (DeParT)** (`jidenn.models.DeParT`).

Sub-modules
-----------
* jidenn.models.BDT
* jidenn.models.DeParT
* jidenn.models.EFN
* jidenn.models.FC
* jidenn.models.Highway
* jidenn.models.PFN
* jidenn.models.ParT
* jidenn.models.Transformer