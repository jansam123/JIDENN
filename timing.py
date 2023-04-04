import tensorflow as tf
import tensorflow_addons as tfa  # this is necessary for the LAMB optimizer to work
import tensorflow_decision_forests as tfdf  # this is necessary for the BDT model to work
import numpy as np
import os
import logging
import hydra
import pandas as pd
import time
from hydra.core.config_store import ConfigStore
#
from src.config import eval_config
from src.data.get_dataset import get_preprocessed_dataset
from src.model_builders.LearningRateSchedulers import LinearWarmup
from src.data.TrainInput import input_classes_lookup


cs = ConfigStore.instance()
cs.store(name="args", node=eval_config.EvalConfig)


@hydra.main(version_base="1.2", config_path="src/config", config_name="eval_config")
def main(args: eval_config.EvalConfig) -> None:
    log = logging.getLogger(__name__)

    if args.seed is not None:
        log.info(f"Setting seed to {args.seed}")
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

    base_model_dir = f'/home/jankovys/JIDENN/good_logs/comparison_12e'
    batch_size = 512
    num_batches = 100
    num_repeats = 10

    custom_objects = {'LinearWarmup': LinearWarmup}
    models1 = ['basic_fc', 'highway']
    models2 = ['depart', 'efn', 'part', 'pfn', 'transformer']
    models3 = ['interacting_depart', 'interacting_part']
    models4 = ['bdt']
    all_models = [models1, models2, models3, models4]

    input_types = ['highlevel', 'constituents', 'interaction_constituents', 'highlevel_constituents']

    df = pd.DataFrame(columns=['model', 'num_params', 'batch_time',
                               'one_jet_time', 'total_time', 'mem_before_current [MB]', 'mem_before_max [MB]', 'mem_after_current [MB]', 'mem_after_max [MB]', 'mem_after_predict_current [MB]', 'mem_after_predict_max [MB]'])

    file = [f'{args.data.path}/{file}/{args.test_subfolder}' for file in args.data.JZ_slices] if args.data.JZ_slices is not None else [
        f'{args.data.path}/{file}/{args.test_subfolder}' for file in os.listdir(args.data.path)]

    file_labels = [int(jz.split('_')[0].lstrip('JZ'))
                   for jz in args.data.JZ_slices] if args.data.JZ_slices is not None else None
    test_ds = get_preprocessed_dataset(file, args.data, file_labels)

    for models, input_type in zip(all_models, input_types):
        train_input_class = input_classes_lookup(input_type)

        train_input_class = train_input_class(per_jet_variables=args.data.variables.perJet,
                                              per_event_variables=args.data.variables.perEvent,
                                              per_jet_tuple_variables=args.data.variables.perJetTuple)

        model_input = tf.function(func=train_input_class)
        ds = test_ds.map_data(model_input)
        ds = ds.get_dataset(batch_size=batch_size, take=num_batches * batch_size)
        ds = ds.cache()
        a = sum(1 for _ in ds)
        check = 1

        for model_name in models:
            model_dir = f'{base_model_dir}/{model_name}/model'
            mem_before = tf.config.experimental.get_memory_info('GPU:0')
            model: tf.keras.Model = tf.keras.models.load_model(model_dir, custom_objects=custom_objects)
            mem_after = tf.config.experimental.get_memory_info('GPU:0')

            if check == 1:
                a = model.predict(ds)
                check = 0

            start = time.time()
            for _ in range(num_repeats):
                a = model.predict(ds)
            end = time.time()

            mem_after_predict = tf.config.experimental.get_memory_info('GPU:0')

            total_inference_time = (end - start) / num_repeats
            batch_inference_time = total_inference_time / num_batches
            one_jet_inference_time = batch_inference_time / batch_size
            num_params = model.count_params()

            new_row = pd.DataFrame([[model_name, num_params, batch_inference_time,
                                    one_jet_inference_time, total_inference_time,
                                    float(mem_before['current']) * 1e-6, float(mem_before['peak']) * 1e-6,
                                    float(mem_after['current']) * 1e-6, float(mem_after['peak']) * 1e-6,
                                    float(mem_after_predict['current']) * 1e-6, float(mem_after_predict['peak']) * 1e-6]],
                                   columns=df.columns)

            print(new_row)
            df = pd.concat([df, new_row], ignore_index=True)
            tf.config.experimental.reset_memory_stats('GPU:0')

    print(df)
    df.to_csv(f'{base_model_dir}/inference_times_cache_{num_repeats}r_{num_batches}nb_{batch_size}bs.csv')


if __name__ == "__main__":
    main()
