import tensorflow as tf
import tensorflow_decision_forests as tfdf
import numpy as np
import os
import logging
import hydra
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from hydra.core.config_store import ConfigStore
#
import src.data.data_info as data_info
from src.config import eval_config
from src.postprocess.pipeline import postprocess_pipe
from src.data.utils.Cut import Cut
from src.data.get_dataset import get_preprocessed_dataset
from src.models.transformer import LinearWarmup
from src.data.get_train_input import get_train_input


cs = ConfigStore.instance()
cs.store(name="args", node=eval_config.EvalConfig)


@hydra.main(version_base="1.2", config_path="src/config", config_name="eval_config")
def main(args: eval_config.EvalConfig) -> None:
    log = logging.getLogger(__name__)

    if args.seed is not None:
        log.info(f"Setting seed to {args.seed}")
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

    file = [f'{args.data.path}/{file}' for file in args.data.JZ_slices] if args.data.JZ_slices is not None else [
        f'{args.data.path}/{file}/train' for file in os.listdir(args.data.path)]

    test = get_preprocessed_dataset(file, args.data)

    args.sub_eval.test_sample_cuts = [] if args.sub_eval.test_sample_cuts is None else args.sub_eval.test_sample_cuts
    sub_tests = [test.filter(lambda x, y, w: Cut(cut)(x['perJet'])) for cut in args.sub_eval.test_sample_cuts]

    custom_objects = {'LinearWarmup': LinearWarmup}
    model: tf.keras.Model = tf.keras.models.load_model(args.model_dir, custom_objects=custom_objects)
    model.summary(print_fn=log.info)

    subsample_results = pd.DataFrame(columns=model.metrics_names+['cut'])
    naming_schema = {0: args.data.labels[0], 1: args.data.labels[1]}

    for cut, dt, name in zip(['base'] + args.sub_eval.test_sample_cuts, [test] + sub_tests, ['base'] + args.sub_eval.test_names):
        dt = dt.map_data(get_train_input(args.model))
        tf_dt = dt.get_dataset(batch_size=args.batch_size,
                               take=args.take)

        sub_test_eval = model.evaluate(tf_dt, return_dict=True)
        if cut != 'base':
            subsample_results = pd.concat([subsample_results, pd.DataFrame({**sub_test_eval, 'cut': name}, index=[0])])
        score = model.predict(tf_dt).ravel()

        dir_name = os.path.join(args.logdir, cut)
        os.makedirs(dir_name, exist_ok=True)
        dist_dir = os.path.join(dir_name, 'dist')
        os.makedirs(dist_dir, exist_ok=True)

        log.info(f"Convert to pandas for cut {cut}")
        df = dt.apply(lambda x: x.take(args.take)).to_pandas()
        if cut == 'base' and args.feature_importance:
            data_info.feature_importance(df, dir_name)
        log.info(f"rename labels")
        df['named_label'] = df['label'].replace(naming_schema)

        if args.draw_distribution is not None:
            data_info.generate_data_distributions(df=df.head(args.draw_distribution)
                                                  if args.draw_distribution != 0 else df, folder=dist_dir)
        log.info(f"Test evaluation for cut {cut}: {sub_test_eval}")

        results_df = pd.DataFrame({'score': score, 'label': df['label'], 'weight': df['weight']})
        results_df['named_label'] = results_df['label'].replace(naming_schema)
        results_df['prediction'] = score.round()
        results_df['prediction'] = results_df['prediction'].astype(int)
        results_df['named_prediction'] = results_df['prediction'].replace(naming_schema)
        postprocess_pipe(results_df, dir_name, log=log)

    # save dataframe to csv
    subsample_results.to_csv(os.path.join(args.logdir, 'results.csv'), index=False)

    # plot the results
    for metric in model.metrics_names:
        log.info(f"Plotting {metric} for cuts")
        sns.pointplot(x='cut', y=metric, data=subsample_results, join=False)
        plt.xlabel('Cut')
        plt.ylabel(metric)
        # put the legend out of the figure
        plt.savefig(os.path.join(args.logdir, f'{metric}.png'))
        plt.close()


if __name__ == "__main__":
    main()
