import tensorflow as tf
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
from src.models.optimizers import LinearWarmup
from src.data.get_train_input import get_train_input
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
    train_input = get_train_input(args.model, interaction=interaction)

    file = [f'{args.data.path}/{file}/{args.test_subfolder}' for file in args.data.JZ_slices] if args.data.JZ_slices is not None else [
        f'{args.data.path}/{file}/{args.test_subfolder}' for file in os.listdir(args.data.path)]
    test_ds = get_preprocessed_dataset(file, args.data)
    args.sub_eval.test_sample_cuts = [] if args.sub_eval.test_sample_cuts is None else args.sub_eval.test_sample_cuts

    for cut, cut_alias in zip(['base'] + args.sub_eval.test_sample_cuts, ['base'] + args.sub_eval.test_names):
        ds = test_ds.filter(lambda x, y, w: Cut(cut)(x['perJet'])) if cut != 'base' else test_ds
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
        df = ds.apply(lambda x: x.take(args.take)).to_pandas()
        if cut == 'base' and args.feature_importance:
            data_info.feature_importance(df, dir_name)
        df['named_label'] = df['label'].replace(naming_schema)

        # calculate metrics
        metrics = calculate_metrics(y_true=df['label'].to_numpy(), y_pred=prediction)
        log.info(f"Test evaluation for cut {cut}: {metrics}")
        if cut != 'base':
            metrics_per_cut = pd.concat([metrics_per_cut, pd.DataFrame({**metrics, 'cut': cut_alias}, index=[0])])

        if args.draw_distribution is not None:
            data_info.generate_data_distributions(df=df.head(args.draw_distribution)
                                                  if args.draw_distribution != 0 else df, folder=dist_dir)

        results_df = pd.DataFrame({'score': score,
                                   'label': df['label'],
                                   'weight': df['weight'],
                                   'named_label': df['named_label'],
                                   'prediction': prediction, })

        results_df['named_prediction'] = results_df['prediction'].replace(naming_schema)
        plot_validation_figs(results_df, dir_name, log=log)

    metrics_per_cut.to_csv(os.path.join(args.logdir, 'results.csv'), index=False)
    plot_metrics_per_cut(metrics_per_cut, args.logdir, log=log)


if __name__ == "__main__":
    main()
