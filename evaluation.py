import tensorflow as tf
import tensorflow_addons as tfa  # this is necessary for the LAMB optimizer to work
import tensorflow_decision_forests as tfdf  # this is necessary for the BDT model to work
import numpy as np
import os
import logging
import hydra
import pandas as pd
from hydra.core.config_store import ConfigStore
#
import src.data.data_info as data_info
from src.config import eval_config
from src.evaluation.plotter import plot_validation_figs, plot_metrics_per_cut
from src.data.utils.Cut import Cut
from src.data.get_dataset import get_preprocessed_dataset
from src.model_builders.LearningRateSchedulers import LinearWarmup
from src.data.get_train_input import get_train_input, get_train_input_old
from src.evaluation.evaluation_metrics import calculate_metrics


cs = ConfigStore.instance()
cs.store(name="args", node=eval_config.EvalConfig)


@hydra.main(version_base="1.2", config_path="src/config", config_name="eval_config")
def main(args: eval_config.EvalConfig) -> None:
    log = logging.getLogger(__name__)

    if args.seed is not None:
        log.info(f"Setting seed to {args.seed}")
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

    custom_objects = {'LinearWarmup': LinearWarmup}
    model: tf.keras.Model = tf.keras.models.load_model(args.model_dir, custom_objects=custom_objects)
    model.summary(print_fn=log.info)

    metrics_per_cut = pd.DataFrame(columns=['cut'])
    naming_schema = {0: args.data.labels[0], 1: args.data.labels[1]}
    interaction = args.interaction if args.model == 'depart' or args.model == 'part' else None
    relative = args.relative if args.model == 'depart' else False

    if args.old:
        train_input = get_train_input_old(args.model, interaction=interaction)
    else:
        train_input = get_train_input(args.model, interaction=interaction, relative=relative)

    file = [f'{args.data.path}/{file}/{args.test_subfolder}' for file in args.data.JZ_slices] if args.data.JZ_slices is not None else [
        f'{args.data.path}/{file}/{args.test_subfolder}' for file in os.listdir(args.data.path)]
    test_ds = get_preprocessed_dataset(file, args.data)
    args.binning.test_sample_cuts = [] if args.binning.test_sample_cuts is None else args.binning.test_sample_cuts

    names = args.binning.test_names if args.binning.test_names is not None else []
    names = ['base'] + names if args.include_base else names
    cuts = args.binning.test_sample_cuts if args.binning.test_sample_cuts is not None else []
    cuts = ['base'] + cuts if args.include_base else cuts

    for cut, cut_alias in zip(cuts, names):
        ds = test_ds.filter(lambda x, y, w: Cut(cut)(x[args.binning.type])) if cut != 'base' else test_ds
        label = ds.get_dataset(batch_size=args.take, take=args.take, map_func=lambda *d: d[1])
        ds = ds.map_data(train_input)
        tf_ds = ds.get_dataset(batch_size=args.batch_size,
                               take=args.take)

        # get model predictions
        score = model.predict(tf_ds).ravel()
        prediction = np.where(score > args.threshold, 1, 0)

        # dir creation
        dir_name = os.path.join(args.logdir, cut_alias)
        os.makedirs(dir_name, exist_ok=True)
        dist_dir = os.path.join(dir_name, 'dist')
        os.makedirs(dist_dir, exist_ok=True)

        # convert to pandas
        log.info(f"Convert to pandas for cut {cut}")
        label = label.as_numpy_iterator().next()
        df = pd.DataFrame({'label': label, 'weight': np.ones_like(label)})
        df['named_label'] = df['label'].replace(naming_schema)

        # calculate metrics
        metrics = calculate_metrics(y_true=df['label'].to_numpy(), score=score)
        log.info(f"Test evaluation for cut {cut}: {metrics}")
        if cut != 'base':
            new = pd.DataFrame({**metrics, 'cut': cut_alias}, index=[0])
            metrics_per_cut = pd.concat([metrics_per_cut, new])
            if not os.path.isfile(os.path.join(args.logdir, 'results.csv')):
                new.to_csv(os.path.join(args.logdir, 'results.csv'), index=False)
            else:
                new.to_csv(os.path.join(args.logdir, 'results.csv'), mode='a', header=False, index=False)

        if args.draw_distribution is not None:
            draw_df = ds.apply(lambda x: x.take(args.draw_distribution)).to_pandas()
            if cut == 'base' and args.feature_importance:
                data_info.feature_importance(draw_df, dir_name)
            draw_df['named_label'] = draw_df['label'].replace(naming_schema)
            data_info.generate_data_distributions(df=draw_df,
                                                  folder=dist_dir,
                                                  color_column='named_label')

        results_df = pd.DataFrame({'score': score,
                                   'label': df['label'],
                                   'weight': df['weight'],
                                   'named_label': df['named_label'],
                                   'prediction': prediction, })

        results_df['named_prediction'] = results_df['prediction'].replace(naming_schema)

        plot_validation_figs(results_df, dir_name, log=log)

    plot_metrics_per_cut(metrics_per_cut, args.logdir, log=log)


if __name__ == "__main__":
    main()
