from dataclasses import dataclass
import tensorflow as tf
import tensorflow_addons as tfa
# this is necessary for the BDT model to work
import numpy as np
import os
import logging
import hydra
import pandas as pd
from multiprocessing import Pool
from hydra.core.config_store import ConfigStore
import seaborn as sns
import ast
#
from jidenn.data.JIDENNDataset import JIDENNDataset, ROOTVariables
from jidenn.config import eval_config
from jidenn.evaluation.plotter import plot_validation_figs, plot_data_distributions, plot_var_dependence
from jidenn.data.get_dataset import get_preprocessed_dataset
from jidenn.model_builders.LearningRateSchedulers import LinearWarmup
from jidenn.evaluation.evaluation_metrics import EffectiveTaggingEfficiency
from jidenn.evaluation.evaluation_metrics import calculate_metrics
from jidenn.evaluation.evaluator import evaluate_multiple_models, calculate_binned_metrics
from utils.const import METRIC_NAMING_SCHEMA, LATEX_NAMING_CONVENTION, MODEL_NAMING_SCHEMA

CUSTOM_OBJECTS = {'LinearWarmup': LinearWarmup,
                  'EffectiveTaggingEfficiency': EffectiveTaggingEfficiency}


cs = ConfigStore.instance()
cs.store(name="args", node=eval_config.EvalConfig)


@hydra.main(version_base="1.2", config_path="jidenn/config", config_name="eval_config")
def main(args: eval_config.EvalConfig) -> None:
    log = logging.getLogger(__name__)

    log.info(f'Using models: {args.model_names}')
    variable = args.binning.variable
    log.info(f'Binning in variable: {variable}')
    labels = args.data.labels

    data_path = [f'{args.data.path}/{jz_slice}/{args.test_subfolder}' for jz_slice in args.data.subfolders] if args.data.subfolders is not None else [
        f'{args.data.path}/{args.test_subfolder}']
    full_dataset = get_preprocessed_dataset(data_path, args.data, None)
    os.makedirs(f'{args.logdir}/data_dist', exist_ok=True)

    def distribution_drawer(x: JIDENNDataset):
        print('Convert to pandas')
        df = x.apply(lambda ds: ds.take(args.take)).to_pandas()
        return plot_data_distributions(df,
                                       folder=f'{args.logdir}/data_dist',
                                       named_labels=labels,
                                       xlabel_mapper=LATEX_NAMING_CONVENTION)
    log.info('Evaluating models')
    full_dataset = evaluate_multiple_models(model_paths=[os.path.join(args.models_path, model_name, 'model') for model_name in args.model_names],
                                            model_names=args.model_names,
                                            model_input_name=args.model_input_types,
                                            dataset=full_dataset,
                                            custom_objects=CUSTOM_OBJECTS,
                                            batch_size=args.batch_size,
                                            log=log,
                                            take=args.take,
                                            distribution_drawer=distribution_drawer)

    variables = [f'{model}_score' for model in args.model_names] + [variable]
    log.info('Converting to pandas')
    df = full_dataset.to_pandas(variables=variables)

    if args.binning.log_bin_base is not None:
        bins = np.logspace(np.log10(args.binning.min_bin), np.log10(args.binning.max_bin),
                           args.binning.bins, base=args.binning.log_bin_base)
    else:
        bins = np.linspace(args.binning.min_bin, args.binning.max_bin, args.binning.bins)

    overall_metrics = pd.DataFrame()

    dfs = []

    for model_name in args.model_names:
        if args.threshold_path is not None and args.threshold_file_name is not None and args.threshold_var_name is not None:
            threshold = pd.read_csv(f'{args.threshold_path}/{model_name}/{args.threshold_file_name}')
        else:
            threshold = 0.5

        log.info(f'Calculating metrics for model: {model_name}')

        os.makedirs(f'{args.logdir}/{model_name}', exist_ok=True)
        plot_validation_figs(df=df[[f'{model_name}_score', 'label']].copy(),
                             logdir=os.path.join(args.logdir, model_name),
                             score_name=f'{model_name}_score',
                             class_names=labels)

        overall_metrics = pd.concat([overall_metrics,
                                    pd.DataFrame(calculate_metrics(df['label'], df[f'{model_name}_score']), index=[model_name])])

        def validation_plotter(x: pd.DataFrame):
            bin_center_name = x['bin'].apply(lambda x: x.mid * 1e-6).iloc[0]
            plot_validation_figs(df=x[[f'{model_name}_score', 'label']].copy(),
                                 logdir=os.path.join(args.logdir, model_name, f'{bin_center_name:.2f}'),
                                 score_name=f'{model_name}_score',
                                 class_names=labels)

        model_df = calculate_binned_metrics(df=df,
                                            binned_variable=variable,
                                            score_variable=f'{model_name}_score',
                                            bins=bins,
                                            validation_plotter=validation_plotter,
                                            threshold=threshold,
                                            threshold_name=args.threshold_var_name,
                                            )

        os.makedirs(f'{args.logdir}/{model_name}', exist_ok=True)

        if variable == 'jets_pt':
            model_df['bin_mid'] = model_df['bin'].apply(lambda x: x.mid * 1e-6)
            model_df['bin_width'] = model_df['bin'].apply(lambda x: x.length * 1e-6)
        else:
            model_df['bin_mid'] = model_df['bin'].apply(lambda x: x.mid)
            model_df['bin_width'] = model_df['bin'].apply(lambda x: x.length)

        model_df.to_csv(f'{args.logdir}/{model_name}/binned_metrics.csv')
        log.info(model_df)
        dfs.append(model_df)

    overall_metrics.to_csv(f'{args.logdir}/overall_metrics.csv')
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
                        xlabel=LATEX_NAMING_CONVENTION[variable],
                        ylabel_mapper=METRIC_NAMING_SCHEMA,
                        save_path=f'{args.logdir}/compare_models',
                        colours=sorted_colours,
                        )


if __name__ == "__main__":
    main()
