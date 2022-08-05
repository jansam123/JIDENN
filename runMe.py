from gc import callbacks
from jidenn_dataset import JIDENNDataset
from models import BasicFCModel, TransformerModel
import numpy as np
import tensorflow as tf
import os
import datetime
from ArgumentParser import ArgumentParser
# import tensorflow_decision_forests as tfdf
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default


def create_dataset(jidenn: JIDENNDataset, args:ArgumentParser, dataset_type:str) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    
    def prep_dataset(data, label, weight):
        label = tf.one_hot(tf.cast(label, tf.int32), JIDENNDataset.LABELS)
        return data, label, weight
    dataset = getattr(jidenn, dataset_type).dataset
    dataset = dataset.map(prep_dataset)
    
    if args.take is not None:
        if dataset_type == 'dev':
            take = int(args.take * args.dev_size)
        elif dataset_type == 'test':
            take = int(args.take * args.test_size)
        else:
            take = args.take
        dataset = dataset.take(take)
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(take))
        
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
    datafiles = [os.path.join(args.data_path, folder, file) for folder in os.listdir(args.data_path) for file in os.listdir(os.path.join(args.data_path, folder)) if '.root' in file]
    np.random.shuffle(datafiles)
    
    jidenn = JIDENNDataset(datafiles, dev_size=args.dev_size, test_size=args.test_size, reading_size=args.reading_size, num_workers=args.num_workers)
    train, dev, test = [create_dataset(jidenn,args, dataset_type) for dataset_type in ["train", "dev", "test"]]

    
    # creating model
    if args.model == "basic_fc":
        model = BasicFCModel(args,len(jidenn.variables), jidenn.LABELS)
    elif args.model == "transformer":
        model = TransformerModel(args,len(jidenn.variables), jidenn.LABELS)
    else:
        assert False, "Model not implemented"
    model.summary()
    
    #callbacks
    callbacks = []
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=args.logdir, update_freq=args.tb_update_freq)
    callbacks += [tb_callback]
    
    #running training
    model.fit(train, validation_data=dev, epochs=args.epochs, callbacks=callbacks, validation_steps=args.validation_batches if args.take is None else None)
        
    
    # print([np.argmax(prediction) for prediction in model.predict(dev.take(1))])


if __name__ == "__main__":
    parser = ArgumentParser()
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
