"""
This moduele contains all the necessary toolkit to:

- convert a ROOT file into a Tensorflow `tf.data.Dataset` object 
- flatten the dataset from events to individal jets
- pick the subsample of desired variables and apply cuts on them
- remap the labels to a classes used as targets in the training
- resample the dataset to balance the classes
- combine datasets created from different ROOT files 
- construct desired input features and targets for the training
- batch and prefetch the dataset for training

These functionalities can be configured using the `jidenn.config` module.

"""