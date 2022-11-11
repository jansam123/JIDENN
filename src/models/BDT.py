import tensorflow as tf
from src.config import config_subclasses as cfg
import tensorflow_decision_forests as tfdf



def create(args_model: cfg.BDT)->tfdf.keras.RandomForestModel:
    
    
    model = tfdf.keras.GradientBoostedTreesModel(
        num_trees=args_model.num_trees,
        growing_strategy=args_model.growing_strategy,
        max_depth=args_model.max_depth,
        split_axis=args_model.split_axis,
        categorical_algorithm=args_model.categorical_algorithm,
        early_stopping='NONE',
        shrinkage=args_model.shrinkage,
        min_examples=args_model.min_examples,
        verbose=2,
        num_threads=args_model.num_threads,
        l2_regularization=args_model.l2_regularization,
        )
    model.compile(
            weighted_metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])
    
    return model

