Module jidenn.models.BDT
========================
BDT model based on `tensorflow_decision_forests` implementation of `tfdf.keras.GradientBoostedTreesModel`.
See https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/GradientBoostedTreesModel for more details.

Functions
---------

    
`bdt_model(args_model: jidenn.config.model_config.BDT) ‑> tensorflow_decision_forests.keras.GradientBoostedTreesModel`
:   Builds a BDT model
    
    Args:
        args_model (cfg_BDT): BDT model config
    
    Returns:
        tfdf.keras.RandomForestModel: BDT model