import tensorflow as tf
import numpy as np
import os
import logging
import hydra
import pandas as pd
from multiprocessing import Pool
from hydra.core.config_store import ConfigStore
import seaborn as sns
from functools import partial
import hashlib
import pickle
import copy

#
from jidenn.data.JIDENNDataset import JIDENNDataset, ROOTVariables
from jidenn.config import eval_config
from jidenn.evaluation.plotter import plot_validation_figs, plot_data_distributions
from jidenn.data.get_dataset import get_preprocessed_dataset
from jidenn.model_builders.LearningRateSchedulers import LinearWarmup
from jidenn.evaluation.evaluation_metrics import EffectiveTaggingEfficiency
from jidenn.evaluation.evaluation_metrics import calculate_metrics
from jidenn.evaluation.WorkingPoint import WorkingPoint
from jidenn.evaluation.evaluator import evaluate_single_model, calculate_binned_metrics
from jidenn.const import METRIC_NAMING_SCHEMA, LATEX_NAMING_CONVENTION, MODEL_NAMING_SCHEMA

CUSTOM_OBJECTS = {'LinearWarmup': LinearWarmup,
                  'EffectiveTaggingEfficiency': EffectiveTaggingEfficiency}


cs = ConfigStore.instance()
cs.store(name="args", node=eval_config.SingleEvalConfig)


@hydra.main(version_base="1.2", config_path="jidenn/yaml_config", config_name="single_eval_config")
def main(args: eval_config.SingleEvalConfig) -> None:
    log = logging.getLogger(__name__)

    # GPU logging
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) == 0:
        log.warning("No GPU found, using CPU")
    for i, gpu in enumerate(gpus):
        gpu_info = tf.config.experimental.get_device_details(gpu)
        log.info(
            f"GPU {i}: {gpu_info['device_name']} with compute capability {gpu_info['compute_capability'][0]}.{gpu_info['compute_capability'][1]}")

    # CUDA logging
    system_config = tf.sysconfig.get_build_info()
    log.info(f"System Config: {system_config}")

    log.info(f'Using model: {args.model_name}')
    binning_variables = args.binning.variable
    log.info(f'Binning in variable: {binning_variables}')
    labels = args.data.labels
    data_path = os.path.join(
        args.data.path, args.test_subfolder) if args.test_subfolder is not None else args.data.path
    args_data = copy.deepcopy(args.data)
    args_data.path = data_path

    try:
        if args.cache_scores is None:
            raise FileNotFoundError
        df = pd.read_csv(os.path.join(args.logdir, args.cache_scores))
        log.info(f'Using cached scores from {args.logdir}/{args.cache_scores}')
        if 'jets_eta' in df.columns:
            df['jets_eta'] = df['jets_eta'].abs()
    except FileNotFoundError:
        log.warning(
            f"No cached scores found at '{args.logdir}/{args.cache_scores}'. Calculating scores.")
        log.info('Evaluating models')
        dataset = get_preprocessed_dataset(args_data=args_data,
                                           input_creator=None, shuffle_reading=False)

        if args.draw_distribution is not None:
            os.makedirs(f'{args.logdir}/data_dist', exist_ok=True)

        def distribution_drawer(x: JIDENNDataset, name: str = None):
            print('Convert to pandas')
            df = x.apply(lambda ds: ds.take(
                args.draw_distribution)).to_pandas()
            df.to_csv(f'{args.logdir}/data_dist/{name}.csv')
            return plot_data_distributions(df,
                                           folder=f'{args.logdir}/data_dist',
                                           named_labels={
                                               i: label for i, label in enumerate(labels)},
                                           weight_variable='weight' if args.data.weight is not None else None,
                                           bins=100,
                                           xlabel_mapper=LATEX_NAMING_CONVENTION)
        score, tech_info = evaluate_single_model(model_path=args.model_path,
                                                 model_name=args.model_name,
                                                 model_input_name=args.model_input_type,
                                                 dataset=dataset,
                                                 custom_objects=CUSTOM_OBJECTS,
                                                 batch_size=args.batch_size,
                                                 log=log,
                                                 take=args.take,
                                                 distribution_drawer=distribution_drawer if args.draw_distribution is not None else None,
                                                 )

        variables = [args.binning.variable] + args.additional_variables
        variables += [args.data.weight] if args.data.weight is not None else []
        log.info('Converting to pandas')
        df = dataset.take(args.take).to_pandas(variables=variables)
        df = df.rename(columns={args.data.weight: 'weight'}
                       ) if args.data.weight is not None else df
        df[f'{args.model_name}_score'] = score
        if 'jets_eta' in df.columns:
            df['jets_eta'] = df['jets_eta'].abs()
        df.to_csv(os.path.join(args.logdir, args.cache_scores)
                  ) if args.cache_scores is not None else None
        pd.Series(tech_info).to_csv(os.path.join(
            args.logdir, 'tech_info.csv')) if args.cache_scores is not None else None
    ##########################################################################################################################################
    log.info('Head of dataframe:')
    log.info(df.head())

    binning_variable = binning_variables if isinstance(
        binning_variables, str) else binning_variables[0]

    if args.binning.log_bin_base is not None:
        min_val = np.log(args.binning.min_bin) / \
            np.log(args.binning.log_bin_base) if args.binning.log_bin_base != 0 else np.log(
                args.binning.min_bin)
        max_val = np.log(args.binning.max_bin) / \
            np.log(args.binning.log_bin_base) if args.binning.log_bin_base != 0 else np.log(
                args.binning.max_bin)
        bins = np.logspace(min_val, max_val,
                           args.binning.bins + 1, base=args.binning.log_bin_base if args.binning.log_bin_base != 0 else np.e)
    elif args.binning.min_bin is not None and args.binning.max_bin is not None:
        bins = np.linspace(args.binning.min_bin,
                           args.binning.max_bin, args.binning.bins + 1)
        df = df[df[binning_variable] < args.binning.max_bin]
        df = df[df[binning_variable] > args.binning.min_bin]
    else:
        bins = np.array(args.binning.bins)
        df = df[df[binning_variable] < bins[-1]]
        df = df[df[binning_variable] > bins[0]]

    print(df)
    log.info(f'Using {len(df)} events for evaluation after binning cuts')

    if args.threads is not None and args.threads > 1 and args.validation_plots_in_bins:
        log.warning(
            'Validation plots in bins are not supported with multithreading. Disabling validation plots in bins.')
        args.validation_plots_in_bins = False

    if args.working_point_path is not None:
        threshold = WorkingPoint.load(os.path.join(args.working_point_path))

        if threshold.binning != args.binning:
            raise ValueError(
                f'Working point binning {threshold.binning} does not match evaluation binning {args.binning}')

    else:
        threshold = 0.5

    log.info(f'Calculating metrics for model: {args.model_name}')

    os.makedirs(f'{args.logdir}/val_figs', exist_ok=True)
    score_df = df[[f'{args.model_name}_score', 'label']].copy() if args.data.weight is None else df[[
        f'{args.model_name}_score', 'label', 'weight']].copy()
    plot_validation_figs(df=score_df,
                         logdir=os.path.join(
                             args.logdir, 'val_figs'),
                         score_name=f'{args.model_name}_score',
                         class_names=labels)

    metrics = pd.DataFrame(calculate_metrics(
        df['label'], df[f'{args.model_name}_score'], weights=df['weight'] if args.data.weight is not None else None), index=[args.model_name])

    # nicely print metrics
    log.info(f'Metrics for model {args.model_name}:')
    log.info(metrics)

    def validation_plotter(x: pd.DataFrame):
        bin_center_name = x['bin'].apply(lambda x: x.mid * 1e-6).iloc[0]
        score_df = x[[f'{args.model_name}_score', 'label']].copy() if args.data.weight is None else df[[
            f'{args.model_name}_score', 'label', 'weight']].copy()
        plot_validation_figs(df=score_df,
                             logdir=os.path.join(
                                 args.logdir, 'val_figs', f'{bin_center_name:.2f}'),
                             score_name=f'{args.model_name}_score',
                             class_names=labels)
    model_df = calculate_binned_metrics(df=df,
                                        binned_variable=binning_variable,
                                        score_variable=f'{args.model_name}_score',
                                        weights_variable=args.data.weight,
                                        bins=bins,
                                        validation_plotter=validation_plotter if args.validation_plots_in_bins else None,
                                        threshold=threshold,
                                        threads=args.threads,
                                        )

    if binning_variables == 'jets_pt':
        model_df['bin_mid'] = model_df['bin'].apply(lambda x: x.mid * 1e-6)
        model_df['bin_width'] = model_df['bin'].apply(
            lambda x: x.length * 1e-6)
    else:
        model_df['bin_mid'] = model_df['bin'].apply(lambda x: x.mid)
        model_df['bin_width'] = model_df['bin'].apply(lambda x: x.length)

    model_df.to_csv(
        f'{args.logdir}/binned_metrics.csv')


    if args.working_point_path is None:
        [WorkingPoint(binning=args.binning, thresholds=model_df[col].values).save(os.path.join(
            args.logdir, f'{col}.pkl')) for col in model_df.columns if col.startswith('threshold')]

    logging.info('DONE')


if __name__ == "__main__":
    main()
