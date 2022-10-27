import src.data.data_info as data_info
from src.config import eval_config
from src.postprocess.pipeline import postprocess_pipe
from src.data.utils.Cut import Cut
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import logging
import hydra
from hydra.core.config_store import ConfigStore
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
    
    test = get_preprocessed_dataset(test_files, args_data=args.data, name="test")
    
    def to_dict(x, y, z):
        sample_dict = {var: x[i] for i, var in enumerate(args.data.variables.perJet)}
        sample_dict.update({var: x[i] for i, var in enumerate(args.data.variables.perEvent)})
        sample_dict.update({args.data.target: y})
        if args.data.weight is not None:
            sample_dict.update({args.data.weight: z})
        return sample_dict
    
    sub_tests = [test.filter(Cut(cut).get_filter_function(to_dict)).prefetch(tf.data.AUTOTUNE)
                 for cut in args.test_sample_cuts]
    test = test.prefetch(tf.data.AUTOTUNE)
    
    model = tf.keras.models.load_model(args.model_dir)
    model.summary(print_fn=log.info)
    
    test_eval = model.evaluate(test, verbose=1, return_dict=True)
    log.info(f"Test evaluation: {test_eval}")
    
    for cut, sub_test in zip(args.test_sample_cuts, sub_tests):
        sub_test_eval = model.evaluate(sub_test, verbose=1, return_dict=True)
        log.info(f"Test evaluation for cut {cut}: {sub_test_eval}")
        
    @tf.function
    def labels_only(x, y, z):
        return y
    
    for cut, dt in zip(['base'] + args.test_sample_cuts, [test] + sub_tests):
        if args.preprocess.draw_distribution is not None:
            log.info(f"Drawing data distribution with {args.preprocess.draw_distribution} samples")
            data_info.generate_data_distributions([train, dev, test], f'{args.params.logdir}/dist',
                                                size=args.preprocess.draw_distribution,
                                                var_names=args.data.variables.perJet+args.data.variables.perEvent,
                                                datasets_names=["train", "dev", "test"])
        dt_labels = dt.map(labels_only)
        dir_name = os.path.join(args.logdir, cut)
        os.makedirs(dir_name, exist_ok=True)
        postprocess_pipe(model, dt, dt_labels, dir_name, log)
    
    
