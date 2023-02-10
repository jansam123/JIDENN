from src.config.model_config import BDT as cfg_BDT
import tensorflow_decision_forests as tfdf


def get_BDT_model(args_model: cfg_BDT)->tfdf.keras.RandomForestModel:
    
    model = tfdf.keras.GradientBoostedTreesModel(
        num_trees=args_model.num_trees,
        growing_strategy=args_model.growing_strategy,
        max_depth=args_model.max_depth,
        split_axis=args_model.split_axis,
        early_stopping='NONE',
        shrinkage=args_model.shrinkage,
        min_examples=args_model.min_examples,
        verbose=2,
        num_threads=args_model.num_threads,
        l2_regularization=args_model.l2_regularization,
        loss='BINOMIAL_LOG_LIKELIHOOD',
        )
    
    return model

