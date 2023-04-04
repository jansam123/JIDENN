import tensorflow as tf
import tensorflow_addons as tfa  # this is necessary for the LAMB optimizer to work
import tensorflow_decision_forests as tfdf  # this is necessary for the BDT model to work
import numpy as np
import os
import logging
import hydra
import pandas as pd
import time
import pickle
from hydra.core.config_store import ConfigStore
#
import src.data.data_info as data_info
from src.config import eval_config
from src.evaluation.plotter import plot_validation_figs, plot_metrics_per_cut
from src.data.utils.Cut import Cut
from src.data.get_dataset import get_preprocessed_dataset
from src.model_builders.LearningRateSchedulers import LinearWarmup
from src.data.TrainInput import input_classes_lookup
from src.evaluation.evaluation_metrics import calculate_metrics
from src.evaluation.variable_latex_names import LATEX_NAMING_CONVENTION


cs = ConfigStore.instance()
cs.store(name="args", node=eval_config.EvalConfig)


@hydra.main(version_base="1.2", config_path="src/config", config_name="eval_config")
def main(args: eval_config.EvalConfig) -> None:
    log = logging.getLogger(__name__)

    if args.seed is not None:
        log.info(f"Setting seed to {args.seed}")
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

    naming_schema = {0: args.data.labels[0], 1: args.data.labels[1]}

    file = [f'{args.data.path}/{file}/{args.test_subfolder}' for file in args.data.JZ_slices] if args.data.JZ_slices is not None else [
        f'{args.data.path}/{file}/{args.test_subfolder}' for file in os.listdir(args.data.path)]
    file_labels = [int(jz.split('_')[0].lstrip('JZ'))
                   for jz in args.data.JZ_slices] if args.data.JZ_slices is not None else None

    test_ds = get_preprocessed_dataset(file, args.data, files_labels=file_labels)
    df = test_ds.apply(lambda x: x.take(100_000)).to_pandas()
    print(df.head(20))
    df['Truth Label'] = df['label'].replace(naming_schema)
    df.to_csv('data_for_plot/dataset_cut.csv', index=False)

    with open('data_for_plot/dataset_cut.pkl', 'wb') as f:
        pickle.dump(df, f)

    for input_type in ['highlevel']: #['highlevel', 'highlevel_constituents', 'constituents', 'relative_constituents', 'interaction_constituents', 'antikt_interaction_constituents', 'deepset_constituents']:
        train_input_class = input_classes_lookup(input_type)
        train_input_class = train_input_class(per_jet_variables=args.data.variables.perJet,
                                              per_event_variables=args.data.variables.perEvent,
                                              per_jet_tuple_variables=args.data.variables.perJetTuple)
        model_input = tf.function(func=train_input_class)
        var_df = test_ds.map_data(model_input).apply(lambda x: x.take(1_000_000)).to_pandas()
        print(var_df.head(20))
        var_df['Truth Label'] = var_df['label'].replace(naming_schema)
        var_df.to_csv(f'data_for_plot/{input_type}_cut.csv', index=False)
        with open(f'data_for_plot/{input_type}_cut.pkl', 'wb') as f:
            pickle.dump(var_df, f)


if __name__ == "__main__":
    main()
