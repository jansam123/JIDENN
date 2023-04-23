"""
BDT model based on `tensorflow_decision_forests` implementation of `tfdf.keras.GradientBoostedTreesModel`.
See https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/GradientBoostedTreesModel for more details.
"""
import tensorflow_decision_forests as tfdf

from jidenn.config import model_config


def bdt_model(args_model: model_config.BDT) -> tfdf.keras.GradientBoostedTreesModel:
    """Builds a BDT model

    Args:
        args_model (cfg_BDT): BDT model config

    Returns:
        tfdf.keras.RandomForestModel: BDT model
    """

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
        max_num_nodes=args_model.max_num_nodes,
        temp_directory=args_model.tmp_dir,
    )

    return model
