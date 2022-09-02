from src.data.jidenn_dataset import JIDENNDataset
from src.models.BasicFCModel import BasicFCModel
from src.models.TransformerModel import TransformerModel
import numpy as np
import tensorflow as tf
import os
import datetime
from src.config.ArgumentParser import ArgumentParser
from src.postprocess.pipeline import postprocess_pipe
import tensorflow_decision_forests as tfdf
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default


def create_dataset(jidenn: JIDENNDataset, args:ArgumentParser, dataset_type:str) -> tf.data.Dataset:
    datasets = getattr(jidenn, dataset_type).datasets
    if len(datasets) == 0:
        raise ValueError("No dataset found")
    conc_dataset=[]
    for dataset in datasets:
        if JIDENNDataset.LABELS > 2:
            def prep_dataset(data, label, weight):
                label = tf.one_hot(tf.cast(label, tf.int32), JIDENNDataset.LABELS)
                return data, label, weight
            dataset = dataset.map(prep_dataset)
        
        if args.take is not None:
            if dataset_type == 'dev':
                take = int(args.take * args.dev_size)
            elif dataset_type == 'test':
                take = int(args.take * args.test_size)
            else:
                take = args.take
            dataset = dataset.take(take)
            # dataset = dataset.apply(tf.data.experimental.assert_cardinality(take))
            conc_dataset.append(dataset)
        
    dataset = tf.data.Dataset.sample_from_datasets(conc_dataset, weights=[0.5,0.5])
    dataset = dataset.apply(tf.data.experimental.assert_cardinality(2*take)) if args.take is not None else dataset
    dataset = dataset.shuffle(buffer_size=args.shuffle_buffer) if args.shuffle_buffer is not None else dataset
    dataset = dataset.batch(args.batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def main(args: ArgumentParser) -> None:
    
    #debug mode for tensorflow
    if args.debug:
        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()

    #fixing seed for reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    # managing threads
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    
    #setting up logs
    args.logdir = os.path.join(args.logdir, "{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d__%H_%M_%S"),
    ))
    os.makedirs(args.logdir, exist_ok=True)
    args.save(os.path.join(args.logdir, "args.json"))
    
    #dataset preparation
    datafiles = [os.path.join(args.data_path, folder, file+':NOMINAL') for folder in os.listdir(args.data_path) for file in os.listdir(os.path.join(args.data_path, folder)) if '.root' in file]
    np.random.shuffle(datafiles)
    
    jidenn = JIDENNDataset(datafiles, dev_size=args.dev_size, test_size=args.test_size, reading_size=args.reading_size, num_workers=args.num_workers)
    train, dev, test = [create_dataset(jidenn,args, dataset_type) for dataset_type in ["train", "dev", "test"]]
    
    
    if args.normalize:
        prep_ds = train.take(args.normalization_size).map(lambda x,y,z:x)
        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(prep_ds)
    else:
        normalizer = None

    # creating model
    if args.model == "basic_fc":
        model = BasicFCModel(args,len(jidenn.variables), jidenn.LABELS, preprocess=normalizer)
        model.summary()   
    elif args.model == "transformer":
        model = TransformerModel(args,len(jidenn.variables), jidenn.LABELS,  preprocess=normalizer)
        model.summary()   
    elif args.model=='BDT':
        model = tfdf.keras.GradientBoostedTreesModel(
            num_trees=args.num_trees,
            growing_strategy=args.growing_strategy,
            max_depth=args.max_depth,
            split_axis=args.split_axis,
            categorical_algorithm=args.categorical_algorithm,
            early_stopping='NONE',
            verbose=2
            )
        model.compile(weighted_metrics=["accuracy"])
    else:
        assert False, "Model not implemented"
    
    #callbacks
    callbacks = []
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=args.logdir, update_freq=args.tb_update_freq, histogram_freq=1, embeddings_freq=1)
    callbacks += [tb_callback]
    
    #running training
    model.fit(train, validation_data=dev, epochs=args.epochs, callbacks=callbacks, validation_steps=args.validation_batches if args.take is None else None)
        
    # model.evaluate(test)
    
    test_dataset = test.unbatch().map(lambda d, l, w: d).batch(args.batch_size)
    test_dataset_labels = test.unbatch().map(lambda x,y,z: y)
    
    postprocess_pipe(model, test_dataset,test_dataset_labels, args)

if __name__ == "__main__":
    parser = ArgumentParser()
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
