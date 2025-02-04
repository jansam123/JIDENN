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
from jidenn.evaluation.plotter import plot_validation_figs, plot_data_distributions, plot_var_dependence
from jidenn.data.get_dataset import get_preprocessed_dataset
from jidenn.model_builders.LearningRateSchedulers import LinearWarmup
from jidenn.evaluation.evaluation_metrics import EffectiveTaggingEfficiency
from jidenn.evaluation.evaluation_metrics import calculate_metrics
from jidenn.evaluation.WorkingPoint import WorkingPoint
from jidenn.evaluation.evaluator import evaluate_multiple_models, calculate_binned_metrics
from jidenn.const import METRIC_NAMING_SCHEMA, LATEX_NAMING_CONVENTION, MODEL_NAMING_SCHEMA

CUSTOM_OBJECTS = {'LinearWarmup': LinearWarmup,
                  'EffectiveTaggingEfficiency': EffectiveTaggingEfficiency}


cs = ConfigStore.instance()
cs.store(name="args", node=eval_config.EvalConfig)


@hydra.main(version_base="1.2", config_path="jidenn/yaml_config", config_name="eval_config")
def main(args: eval_config.EvalConfig) -> None:
    config_hash = hashlib.sha256(str(args).encode('utf-8')).hexdigest()
    checkpoint_path = os.path.join(args.logdir, 'tmp', config_hash)
    os.makedirs(checkpoint_path, exist_ok=True)
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

    log.info(f'Using models: {args.model_names}')
    binning_variable = args.binning.variable
    additional_variables = args.additional_variables
    log.info(f'Binning in variable: {binning_variable}')
    log.info(f'Additional variables: {additional_variables}')
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
        full_dataset, tech_info = evaluate_multiple_models(model_paths=[os.path.join(args.models_path, model_name, 'model') for model_name in args.model_names],
                                                           model_names=args.model_names,
                                                           model_input_name=args.model_input_types,
                                                           dataset=dataset,
                                                           custom_objects=CUSTOM_OBJECTS,
                                                           batch_size=args.batch_size,
                                                           log=log,
                                                           take=args.take,
                                                           checkpoint_path=checkpoint_path,
                                                           distribution_drawer=distribution_drawer if args.draw_distribution is not None else None,
                                                           )

        variables = [f'{model}_score' for model in args.model_names]
        variables += [binning_variable] + additional_variables
        variables += [args.data.weight] if args.data.weight is not None else []
        log.info('Converting to pandas')
        df = full_dataset.to_pandas(variables=variables)
        df = df.rename(columns={args.data.weight: 'weight'}
                       ) if args.data.weight is not None else df
        if 'jets_eta' in df.columns:
            df['jets_eta'] = df['jets_eta'].abs()
        df.to_csv(os.path.join(args.logdir, args.cache_scores)
                  ) if args.cache_scores is not None else None
        tech_info.to_csv(os.path.join(args.logdir, 'tech_info.csv'))
        tech_info.index.name = 'Model'
        tech_info = tech_info.reset_index()
        tech_info.to_latex(buf=f'{args.logdir}/tech_info.tex', float_format="{:.2f}".format,
                           column_format='l' + 'cc', label='tab:tech_info', index=False,
                           escape=False)
    ##########################################################################################################################################


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
    dfs = []
    overall_metrics = []

    if args.threads is not None and args.threads > 1 and args.validation_plots_in_bins:
        log.warning(
            'Validation plots in bins are not supported with multithreading. Disabling validation plots in bins.')
        args.validation_plots_in_bins = False

    for model_name in args.model_names:
        try:
            with open(os.path.join(checkpoint_path, f'{model_name}.pkl'), 'rb') as f:
                model_df = pickle.load(f)
                dfs.append(model_df)
            with open(os.path.join(checkpoint_path, f'{model_name}_overall_metrics.pkl'), 'rb') as f:
                metrics = pickle.load(f)
                overall_metrics.append(metrics)
            log.info(
                f'Loaded binned metrics for model {model_name} from cache')
            continue
        except FileNotFoundError:
            pass

        if args.working_point_path is not None and args.working_point_file_name is not None:
            # threshold = pd.read_csv(f'{args.threshold_path}/{model_name}/{args.threshold_file_name}')
            threshold = WorkingPoint.load(os.path.join(args.working_point_path,
                                          model_name, args.working_point_file_name))

            if threshold.binning != args.binning:
                raise ValueError(
                    f'Working point binning {threshold.binning} does not match evaluation binning {args.binning}')

        else:
            threshold = 0.5

        log.info(f'Calculating metrics for model: {model_name}')

        os.makedirs(f'{args.logdir}/models/{model_name}', exist_ok=True)
        score_df = df[[f'{model_name}_score', 'label']].copy() if args.data.weight is None else df[[
            f'{model_name}_score', 'label', 'weight']].copy()
        plot_validation_figs(df=score_df,
                             logdir=os.path.join(
                                 args.logdir, 'models', model_name),
                             score_name=f'{model_name}_score',
                             class_names=labels)

        metrics = pd.DataFrame(calculate_metrics(
            df['label'], df[f'{model_name}_score'], weights=df['weight'] if args.data.weight is not None else None), index=[model_name])
        overall_metrics.append(metrics)

        # nicely print metrics
        log.info(f'Metrics for model {model_name}:')
        log.info(metrics)

        def validation_plotter(x: pd.DataFrame):
            bin_center_name = x['bin'].apply(lambda x: x.mid * 1e-6).iloc[0]
            score_df = x[[f'{model_name}_score', 'label']].copy() if args.data.weight is None else df[[
                f'{model_name}_score', 'label', 'weight']].copy()
            plot_validation_figs(df=score_df,
                                 logdir=os.path.join(
                                     args.logdir, 'models', model_name, f'{bin_center_name:.2f}'),
                                 score_name=f'{model_name}_score',
                                 class_names=labels)
        model_df = evaluation(df=df,
                                            binned_variable=binning_variable,
                                            score_variable=f'{model_name}_score',
                                            weights_variable=args.data.weight,
                                            bins=bins,
                                            validation_plotter=validation_plotter if args.validation_plots_in_bins else None,
                                            threshold=threshold,
                                            threads=args.threads,
                                            )

        os.makedirs(f'{args.logdir}/models/{model_name}', exist_ok=True)

        if binning_variables == 'jets_pt':
            model_df['bin_mid'] = model_df['bin'].apply(lambda x: x.mid * 1e-6)
            model_df['bin_width'] = model_df['bin'].apply(
                lambda x: x.length * 1e-6)
        else:
            model_df['bin_mid'] = model_df['bin'].apply(lambda x: x.mid)
            model_df['bin_width'] = model_df['bin'].apply(lambda x: x.length)

        model_df.to_csv(
            f'{args.logdir}/models/{model_name}/binned_metrics.csv')

        if args.working_point_path is None and args.working_point_file_name is None:
            for col in model_df.columns:
                if col.startswith('threshold'):
                    wp = WorkingPoint(binning=args.binning,
                                      thresholds=model_df[col].values)
                    wp.save(os.path.join(args.logdir, 'models',
                            model_name, f'{col}.pkl'))

        with open(os.path.join(checkpoint_path, f'{model_name}.pkl'), 'wb') as f:
            pickle.dump(model_df, f)
        with open(os.path.join(checkpoint_path, f'{model_name}_overall_metrics.pkl'), 'wb') as f:
            pickle.dump(metrics, f)

        dfs.append(model_df)

    overall_metrics = pd.concat(overall_metrics)
    overall_metrics.to_csv(f'{args.logdir}/overall_metrics.csv')
    latex_metrics = overall_metrics[['binary_accuracy', 'auc', 'gluon_efficiency',
                                     'quark_efficiency', 'gluon_rejection_at_quark_50wp', 'gluon_rejection_at_quark_80wp']]
    latex_metrics.index.name = 'Model'
    latex_metrics = latex_metrics.rename(
        columns=METRIC_NAMING_SCHEMA, index=MODEL_NAMING_SCHEMA)
    latex_metrics = latex_metrics.reset_index()
    latex_metrics.to_latex(buf=f'{args.logdir}/overall_metrics.tex', float_format="{:.4f}".format,
                           column_format='l' + 'c' * (len(latex_metrics.columns) - 1), label='tab:results', index=False,
                           escape=False)
    log.info('Overall metrics for all models:')
    log.info(overall_metrics)

    os.makedirs(f'{args.logdir}/compare_models', exist_ok=True)
    acc_sorted_models = overall_metrics.sort_values(
        by='binary_accuracy', ascending=False).index
    colours = sns.color_palette("coolwarm", len(args.model_names))
    sorted_colours = [colours[acc_sorted_models.get_loc(
        model)] for model in args.model_names]
    log.info(f'Plotting variable dependence for models: {acc_sorted_models}')

    labels = []
    for model in args.model_names:
        if model in acc_sorted_models and model in MODEL_NAMING_SCHEMA:
            labels.append(MODEL_NAMING_SCHEMA[model])
        elif model in acc_sorted_models:
            labels.append(model)
        else:
            continue
    if args.reference_model is not None and args.reference_model in acc_sorted_models and args.reference_model in MODEL_NAMING_SCHEMA:
        args.reference_model = MODEL_NAMING_SCHEMA[args.reference_model]
    elif args.reference_model is not None and args.reference_model in acc_sorted_models:
        args.reference_model = args.reference_model
    else:
        args.reference_model = None

    plot_var_dependence(dfs=dfs,
                        labels=labels,
                        ratio_reference_label=args.reference_model,
                        bin_midpoint_name='bin_mid',
                        bin_width_name='bin_width',
                        n_counts='eff_num_events',
                        metric_names=args.metrics_to_plot,
                        ylims=args.ylims,
                        xlog=args.binning.log_bin_base is not None,
                        xlabel=LATEX_NAMING_CONVENTION[binning_variable],
                        ylabel_mapper=METRIC_NAMING_SCHEMA,
                        save_path=f'{args.logdir}/compare_models',
                        colours=sorted_colours,
                        figsize=(10, 10),
                        leg_fontsize=12,
                        )
    logging.info('DONE')
    # remove whole tmp dir
    os.system(f'rm -rf {checkpoint_path}')


if __name__ == "__main__":
    main()
