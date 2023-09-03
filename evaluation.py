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
    variable = args.binning.variable
    log.info(f'Binning in variable: {variable}')
    labels = args.data.labels
    data_path = os.path.join(args.data.path, args.test_subfolder)

    try:
        if args.cache_scores is None:
            raise FileNotFoundError
        df = pd.read_csv(os.path.join(args.logdir, args.cache_scores))
        log.info(f'Using cached scores from {args.logdir}/{args.cache_scores}')
    except FileNotFoundError:
        log.warning(f"No cached scores found at '{args.logdir}/{args.cache_scores}'. Calculating scores.")
        log.info('Evaluating models')
        dataset = get_preprocessed_dataset(file=data_path, args_data=args.data,
                                           input_creator=None, shuffle_reading=False)

        if args.draw_distribution is not None:
            os.makedirs(f'{args.logdir}/data_dist', exist_ok=True)

        def distribution_drawer(x: JIDENNDataset):
            print('Convert to pandas')
            df = x.apply(lambda ds: ds.take(args.draw_distribution)).to_pandas()
            return plot_data_distributions(df,
                                           folder=f'{args.logdir}/data_dist',
                                           named_labels={i: label for i, label in enumerate(labels)},
                                           weight_variable='weight' if args.data.weight is not None else None,
                                           bins=100,
                                           xlabel_mapper=LATEX_NAMING_CONVENTION)
        full_dataset = evaluate_multiple_models(model_paths=[os.path.join(args.models_path, model_name, 'model') for model_name in args.model_names],
                                                model_names=args.model_names,
                                                model_input_name=args.model_input_types,
                                                dataset=dataset,
                                                custom_objects=CUSTOM_OBJECTS,
                                                batch_size=args.batch_size,
                                                log=log,
                                                take=args.take,
                                                distribution_drawer=distribution_drawer if args.draw_distribution is not None else None,
                                                )
        
        variables = [f'{model}_score' for model in args.model_names] + [variable]
        variables += [args.data.weight] if args.data.weight is not None else []
        log.info('Converting to pandas')
        df = full_dataset.to_pandas(variables=variables)
        df = df.rename(columns={args.data.weight: 'weight'}) if args.data.weight is not None else df
        df.to_csv(os.path.join(args.logdir, args.cache_scores)) if args.cache_scores is not None else None

    if args.binning.log_bin_base is not None:
        min_val = np.log(args.binning.min_bin) / \
            np.log(args.binning.log_bin_base) if args.binning.log_bin_base != 0 else np.log(args.binning.min_bin)
        max_val = np.log(args.binning.max_bin) / \
            np.log(args.binning.log_bin_base) if args.binning.log_bin_base != 0 else np.log(args.binning.max_bin)
        bins = np.logspace(min_val, max_val,
                           args.binning.bins + 1, base=args.binning.log_bin_base if args.binning.log_bin_base != 0 else np.e)
    else:
        bins = np.linspace(args.binning.min_bin, args.binning.max_bin, args.binning.bins + 1)

    df = df[df[args.binning.variable] < args.binning.max_bin]
    df = df[df[args.binning.variable] > args.binning.min_bin]
    log.info(f'Using {len(df)} events for evaluation after binning cuts')
    overall_metrics = pd.DataFrame()
    dfs = []

    if args.threads is not None and args.threads > 1 and args.validation_plots_in_bins:
        log.warning('Validation plots in bins are not supported with multithreading. Disabling validation plots in bins.')
        args.validation_plots_in_bins = False

    for model_name in args.model_names:

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
                             logdir=os.path.join(args.logdir, 'models', model_name),
                             score_name=f'{model_name}_score',
                             class_names=labels)

        overall_metrics: pd.DataFrame = pd.concat([overall_metrics,
                                                  pd.DataFrame(calculate_metrics(df['label'], df[f'{model_name}_score'], weights=df['weight'] if args.data.weight is not None else None), index=[model_name])])
        # nicely print metrics
        log.info(f'Metrics for model {model_name}:')
        log.info(''.join([f'{metric_name} = {metric:.4}\n' for metric_name,
                 metric in dict(overall_metrics.loc[model_name]).items()]))

        def validation_plotter(x: pd.DataFrame):
            bin_center_name = x['bin'].apply(lambda x: x.mid * 1e-6).iloc[0]
            score_df = x[[f'{model_name}_score', 'label']].copy() if args.data.weight is None else df[[
                f'{model_name}_score', 'label', 'weight']].copy()
            plot_validation_figs(df=score_df,
                                 logdir=os.path.join(args.logdir, 'models', model_name, f'{bin_center_name:.2f}'),
                                 score_name=f'{model_name}_score',
                                 class_names=labels)
        model_df = calculate_binned_metrics(df=df,
                                            binned_variable=args.binning.variable,
                                            score_variable=f'{model_name}_score',
                                            weights_variable=args.data.weight,
                                            bins=bins,
                                            validation_plotter=validation_plotter if args.validation_plots_in_bins else None,
                                            threshold=threshold,
                                            threads=args.threads,
                                            )

        os.makedirs(f'{args.logdir}/models/{model_name}', exist_ok=True)

        if variable == 'jets_pt':
            model_df['bin_mid'] = model_df['bin'].apply(lambda x: x.mid * 1e-6)
            model_df['bin_width'] = model_df['bin'].apply(lambda x: x.length * 1e-6)
        else:
            model_df['bin_mid'] = model_df['bin'].apply(lambda x: x.mid)
            model_df['bin_width'] = model_df['bin'].apply(lambda x: x.length)

        model_df.to_csv(f'{args.logdir}/models/{model_name}/binned_metrics.csv')

        if args.working_point_path is None and args.working_point_file_name is None:
            for col in model_df.columns:
                if col.startswith('threshold'):
                    wp = WorkingPoint(binning=args.binning, thresholds=model_df[col].values)
                    wp.save(os.path.join(args.logdir, 'models', model_name, f'{col}.pkl'))

        dfs.append(model_df)

    overall_metrics.to_csv(f'{args.logdir}/overall_metrics.csv')
    latex_metrics = overall_metrics[['binary_accuracy', 'auc', 'gluon_efficiency',
                                     'quark_efficiency', 'gluon_rejection_at_quark_50wp', 'gluon_rejection_at_quark_80wp']]
    latex_metrics.index.name = 'Model'
    latex_metrics = latex_metrics.rename(columns=METRIC_NAMING_SCHEMA, index=MODEL_NAMING_SCHEMA)
    latex_metrics = latex_metrics.reset_index()
    latex_metrics.to_latex(buf=f'{args.logdir}/overall_metrics.tex', float_format="{:.4f}".format,
                           column_format='l' + 'c' * (len(latex_metrics.columns) - 1), label='results', index=False,
                           escape=False, caption="Results of the different models. The best results are highlighted in bold.")
    log.info('Overall metrics for all models:')
    log.info(overall_metrics)

    os.makedirs(f'{args.logdir}/compare_models', exist_ok=True)
    acc_sorted_models = overall_metrics.sort_values(by='binary_accuracy', ascending=False).index
    colours = sns.color_palette("coolwarm", len(args.model_names))
    sorted_colours = [colours[acc_sorted_models.get_loc(model)] for model in args.model_names]
    log.info(f'Plotting variable dependence for models: {acc_sorted_models}')
    plot_var_dependence(dfs=dfs,
                        labels=[MODEL_NAMING_SCHEMA[model] for model in args.model_names if model in acc_sorted_models],
                        ratio_reference_label=MODEL_NAMING_SCHEMA[
                            args.reference_model] if args.reference_model and args.reference_model in acc_sorted_models is not None else None,
                        bin_midpoint_name='bin_mid',
                        bin_width_name='bin_width',
                        metric_names=args.metrics_to_plot,
                        ylims=args.ylims,
                        xlog=args.binning.log_bin_base is not None,
                        xlabel=LATEX_NAMING_CONVENTION[variable],
                        ylabel_mapper=METRIC_NAMING_SCHEMA,
                        save_path=f'{args.logdir}/compare_models',
                        colours=sorted_colours,
                        )
    logging.info('DONE')


if __name__ == "__main__":
    main()
