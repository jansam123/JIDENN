import tensorflow as tf
import numpy as np
import pandas as pd
import os
import logging
import hydra
from hydra.core.config_store import ConfigStore
#
import src.data.data_info as data_info
from src.config import eval_config
from src.postprocess.pipeline import postprocess_pipe
from src.data.utils.Cut import Cut
from src.data.get_dataset import get_preprocessed_dataset


cs = ConfigStore.instance()
cs.store(name="args", node=eval_config.EvalConfig)


@hydra.main(version_base="1.2", config_path="src/config", config_name="eval_config")
def main(args: eval_config.EvalConfig) -> None:
    log = logging.getLogger(__name__)

    if args.seed is not None:
        log.info(f"Setting seed to {args.seed}")
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)
        
    folders = os.listdir(args.data.path)
    folders.sort()
    slices = args.data.JZ_slices if args.data.JZ_slices is not None else list(range(1, 13))
    folders = [folders[slc-1] for slc in slices]
    log.info(f"Folders used for evaluation: {folders}")
     
    test_files = [[os.path.join(args.data.path, folder, args.test_subfolder, file)
                  for file in os.listdir(os.path.join(args.data.path, folder, args.test_subfolder))] for folder in folders]
    log.info(f"Files used for evaluation: {test_files}")
    
    test = get_preprocessed_dataset(test_files, args_data=args.data, name="test", size=args.take)
    
    def to_dict(x, y, z):
        sample_dict = {var: x[i] for i, var in enumerate(args.data.variables.perJet)}
        sample_dict.update({var: x[i] for i, var in enumerate(args.data.variables.perEvent)})
        sample_dict.update({args.data.target: y})
        if args.data.weight is not None:
            sample_dict.update({args.data.weight: z})
        return sample_dict
    
    args.test_sample_cuts = [] if args.test_sample_cuts is None else args.test_sample_cuts
    sub_tests = [test.filter(Cut(cut).get_filter_function(to_dict)).batch(args.batch_size).prefetch(tf.data.AUTOTUNE) for cut in args.test_sample_cuts] 
    test = test.batch(args.batch_size)
    
    test = test.prefetch(tf.data.AUTOTUNE)
    
    
    model = tf.keras.models.load_model(args.model_dir)
    model.summary(print_fn=log.info)
    
        
    
    for cut, dt in zip(['base'] + args.test_sample_cuts, [test] + sub_tests):
        dir_name = os.path.join(args.logdir, cut)
        os.makedirs(dir_name, exist_ok=True)
        dist_dir = os.path.join(dir_name, 'dist')
        os.makedirs(dist_dir, exist_ok=True)
        
        df = data_info.tf_dataset_to_pandas(dataset=dt.unbatch(), var_names=args.data.variables.perJet+args.data.variables.perEvent)
        #!
        # if cut == 'base':
        #     data_info.feature_importance(df, dir_name)
        # data_info.generate_data_distributions(df=df, folder=dist_dir)                                                 
        # sub_test_eval = model.evaluate(dt, verbose=1, return_dict=True)
        # log.info(f"Test evaluation for cut {cut}: {sub_test_eval}")
        # dt_labels = dt.unbatch().map(labels_only)
        #!
        score = model.predict(dt).ravel()
        df['score'] = score
        df['prediction'] = score.round()
        df['named_label'] = df['label'].replace({0: args.data.labels[0], 1: args.data.labels[1]})
        df['named_prediction'] = df['prediction'].replace({0: args.data.labels[0], 1: args.data.labels[1]})
        postprocess_pipe(df, dir_name, log=log)
    

if __name__ == "__main__":
    main()