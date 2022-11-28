import tensorflow as tf
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
from src.data.utils.CutV2 import Cut
from src.data.get_dataset import get_preprocessed_dataset
from src.models.transformer import LinearWarmup
from src.data.JIDENNDatasetV2 import get_constituents, get_high_level_variables


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


    args.test_sample_cuts = [] if args.test_sample_cuts is None else args.test_sample_cuts
    sub_tests = [test.filter(lambda x, y, w: Cut(cut)(x['perJet'])) for cut in args.test_sample_cuts]

    custom_objects = {'LinearWarmup': LinearWarmup}
    model: tf.keras.Model = tf.keras.models.load_model(args.model_dir, custom_objects=custom_objects)
    model.summary(print_fn=log.info)

    subsample_results = pd.DataFrame(columns=model.metrics_names+['cut'])
    naming_schema = {0: args.data.labels[0], 1: args.data.labels[1]}
    model_to_data_mapping = {
        "basic_fc": get_high_level_variables,
        "transformer": get_constituents,
        "highway": get_high_level_variables,
        "BDT": get_high_level_variables,
    }
    for cut, dt in zip(['base'] + args.test_sample_cuts, [test] + sub_tests):
        
        tf_dt = dt.get_dataset(batch_size=args.batch_size,
                               map_func=model_to_data_mapping[args.model], 
                               take=args.take)

        sub_test_eval = model.evaluate(tf_dt, return_dict=True)
        subsample_results = pd.concat([subsample_results, pd.DataFrame({**sub_test_eval, 'cut': cut}, index=[0])])
        score = model.predict(tf_dt).ravel()

        dir_name = os.path.join(args.logdir, cut)
        os.makedirs(dir_name, exist_ok=True)
        dist_dir = os.path.join(dir_name, 'dist')
        os.makedirs(dist_dir, exist_ok=True)

        df = dt.apply(lambda x: x.take(args.take)).to_pandas()
        if cut == 'base' and args.feature_importance:
            data_info.feature_importance(df, dir_name)
        if 'perJetTuple.jets_PFO_pt' in df.columns:
            df['perJet.jets_PFO_n'] = df['perJetTuple.jets_PFO_pt'].apply(lambda x: len(x))
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

    subsample_results = subsample_results.reset_index(drop=True)
    subsample_results.index.name = 'id'
    subsample_results = subsample_results.reset_index()
    subsample_results['name'] = subsample_results['id'].astype(str) + ' ' + subsample_results['cut']
    for metric in model.metrics_names:
        log.info(f"Plotting {metric} for cuts")
        sns.pointplot(x='id', y='loss', data=subsample_results, hue='name', join=False)
        plt.xlabel('Cut')
        plt.ylabel(metric)
        plt.savefig(os.path.join(args.logdir, f'{metric}.png'))
        plt.close()


if __name__ == "__main__":
    main()
