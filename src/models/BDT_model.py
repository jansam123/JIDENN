import tensorflow as tf
from ..config.ArgumentParser import ArgumentParser
import tensorflow_decision_forests as tfdf



def create(args: ArgumentParser)->tfdf.keras.RandomForestModel:
    
    if args.epochs != 1:
        print("WARNING: BDTModel only supports 1 epoch. Setting epochs to 1")
        args.epochs = 1
    
    model = tfdf.keras.GradientBoostedTreesModel(
        num_trees=args.num_trees,
        growing_strategy=args.growing_strategy,
        max_depth=args.max_depth,
        split_axis=args.split_axis,
        categorical_algorithm=args.categorical_algorithm,
        early_stopping='NONE',
        verbose=2
        )
    model.compile(
            weighted_metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])
    
    return model

