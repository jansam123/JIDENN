

import tensorflow as tf
import numpy as np
import os
import logging
import hydra
import pandas as pd
from hydra.core.config_store import ConfigStore
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
from jidenn.evaluation.binning_config import BINNINGS
from jidenn.const import METRIC_NAMING_SCHEMA, LATEX_NAMING_CONVENTION, MODEL_NAMING_SCHEMA

CUSTOM_OBJECTS = {'LinearWarmup': LinearWarmup,
                  'EffectiveTaggingEfficiency': EffectiveTaggingEfficiency}


cs = ConfigStore.instance()
cs.store(name="args", node=eval_config.SingleEvalConfig)


@hydra.main(version_base="1.2", config_path="jidenn/yaml_config", config_name="single_eval_config")
def main(args: eval_config.SingleEvalConfig) -> None:
    log = logging.getLogger(__name__)
    # theck if file f'{args.logdir}/binned_metrics.csv' exists, if so, return
    if os.path.isfile(f'{args.logdir}/binned_metrics.csv'):
        log.warning(f'File {args.logdir}/binned_metrics.csv already exists. Skipping evaluation.')
        return
    

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
    # binning_variables = args.binning.variable
    # log.info(f'Binning in variable: {binning_variables}')
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
                                           named_labels={i: label for i, label in enumerate(labels)},
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

        variables = args.include_variables
        log.info('Converting to pandas')
        df = dataset.take(args.take) if args.take is not None and args.take > 0 else dataset
        df = df.to_pandas(variables=variables)
        # df = df.rename(columns={args.data.weight: 'weight'}
        #                ) if args.data.weight is not None else df
        df[f'{args.model_name}_score'] = score
        if 'jets_eta' in df.columns:
            df['jets_eta'] = df['jets_eta'].abs()
        df.to_csv(os.path.join(args.logdir, args.cache_scores)
                  ) if args.cache_scores is not None else None
        pd.Series(tech_info).to_csv(os.path.join(
            args.logdir, 'tech_info.csv')) if args.cache_scores is not None else None
    ##########################################################################################################################################
    log.info('Head of dataframe:')
    if not 'jets_Constituent_n+jets_TopoTower_n' in df.columns and 'jets_Constituent_n' in df.columns and 'jets_TopoTower_n' in df.columns:
        df['jets_Constituent_n+jets_TopoTower_n'] = df['jets_Constituent_n'] + df['jets_TopoTower_n']
    log.info(df.head())

    # log.info(f'Using {len(df)} events for evaluation after binning cuts')

    if args.threads is not None and args.threads > 1 and args.validation_plots_in_bins:
        log.warning(
            'Validation plots in bins are not supported with multithreading. Disabling validation plots in bins.')
        args.validation_plots_in_bins = False


    log.info(f'Calculating metrics for model: {args.model_name}')
    log.info(df)
    log.info(df.columns)

    if not os.path.isfile(os.path.join(args.logdir, 'overall_metrics.csv')):
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
        # log.info(metrics)
        metrics.to_csv(os.path.join(args.logdir, 'overall_metrics.csv'))
    else:
        log.warning(
            f"File {os.path.join(args.logdir, 'overall_metrics.csv')} already exists. Skipping evaluation.")
        
    os.makedirs(f'{args.logdir}/binned_metrics', exist_ok=True)
    
    for binning in BINNINGS:
        log.info(f'Binning in variable {binning.variable} ')
        
        bins = np.array(binning.bins)
        binning_variable = binning.variable
        loc_df = df[df[binning_variable] < bins[-1]]
        loc_df = df[df[binning_variable] > bins[0]]
        wp_binning = eval_config.BinningConfig(variable=binning_variable, bins=binning.bins, max_bin=None, min_bin=None, log_bin_base=None)
        
        if isinstance(binning.name, str):
            cuts = [binning.cut]
            cut_names = [binning.name]
        elif binning.name is None:
            cuts = [None]
            cut_names = [None]
        else:
            cuts = binning.cut  
            cut_names = binning.name
        
        for cut, cut_name in zip(cuts, cut_names): 
            if os.path.isfile(os.path.join(args.logdir, 'binned_metrics', f'{binning_variable}.csv' if cut_name is None else f'{binning_variable}_{cut_name}.csv')):
                log.warning(f"File {os.path.join(args.logdir, 'binned_metrics', f'{binning_variable}.csv' if cut_name is None else f'{binning_variable}_{cut_name}.csv')} already exists. Skipping evaluation.")
                continue
            log.info(f'Cut: {cut_name}')
            if cut is not None:
                cut_loc_df = loc_df.query(cut)
                log.info(f'Num jets after cut: {len(cut_loc_df)}')
            else:
                cut_loc_df = loc_df.copy()
                log.info(f'Num jets (no cut specified): {len(cut_loc_df)}')
        
            if args.working_point_path is not None:
                if args.working_point_file is None:
                    raise ValueError('Both `working_point_path` and `working_point_file` must be set.')
                
                threshold = WorkingPoint.load(os.path.join(args.working_point_path, f'{binning_variable}' if cut_name is None else f'{binning_variable}_{cut_name}', args.working_point_file))
                logging.info(f'Using working point, by loading threshold from {args.working_point_path}')

                if threshold.binning != wp_binning:
                    raise ValueError(
                        f'Working point binning {threshold.binning} does not match evaluation binning {wp_binning}')

            else:
                threshold = 0.5
                logging.info(f'No working point path given, using threshold {threshold}')
            
            def validation_plotter(x: pd.DataFrame):
                if binning.variable == 'jets_pt':
                    bin_left= x['bin'].apply(lambda x: x.left * 1e-3).iloc[0]
                    bin_right = x['bin'].apply(lambda x: x.right * 1e-3).iloc[0]
                    name = f'{bin_left:.0f}-{bin_right:.0f} GeV'
                else:
                    bin_left = x['bin'].apply(lambda x: x.left).iloc[0]
                    bin_right = x['bin'].apply(lambda x: x.right).iloc[0]
                    name = f'{bin_left:.2f}-{bin_right:.2f}'
                    
                plot_validation_figs(df=x[[f'{args.model_name}_score', 'label']].copy() if args.data.weight is None else x[[f'{args.model_name}_score', 'label', 'weight']].copy(),
                                    logdir=os.path.join(
                                        args.logdir, 'val_figs', f'{name}'),
                                    score_name=f'{args.model_name}_score',
                                    subtext=name,
                                    class_names=labels)
                
            log.info(f'Max jets pt: {cut_loc_df["jets_pt"].max()}')
            log.info(f'Min jets pt: {cut_loc_df["jets_pt"].min()}')
            model_df = calculate_binned_metrics(df=cut_loc_df,
                                                binned_variable=binning_variable,
                                                score_variable=f'{args.model_name}_score',
                                                weights_variable='weight' if args.data.weight is not None else None,
                                                bins=bins,
                                                validation_plotter=validation_plotter if args.validation_plots_in_bins else None,
                                                threshold=threshold,
                                                logger=log,
                                                )
            log.info(model_df)
            
            if binning_variable == 'jets_pt':
                model_df['bin_mid'] = model_df['bin'].apply(lambda x: x.mid * 1e-3)
                model_df['bin_width'] = model_df['bin'].apply(
                    lambda x: x.length * 1e-3)
            else:
                model_df['bin_mid'] = model_df['bin'].apply(lambda x: x.mid)
                model_df['bin_width'] = model_df['bin'].apply(lambda x: x.length)

            model_df.to_csv(os.path.join(args.logdir, 'binned_metrics', f'{binning_variable}.csv' if cut_name is None else f'{binning_variable}_{cut_name}.csv'))

            if args.working_point_path is None:
                wp_save_path = os.path.join(args.logdir, 'threshold', f'{binning_variable}' if cut_name is None else f'{binning_variable}_{cut_name}')
                os.makedirs(wp_save_path, exist_ok=True)
                [WorkingPoint(binning=wp_binning, thresholds=model_df[col].values).save(os.path.join(wp_save_path, f'{col}.pkl')) for col in model_df.columns if col.startswith('threshold')]
                
            log.info('')
        log.info('')

    logging.info('DONE')


if __name__ == "__main__":
    main()
