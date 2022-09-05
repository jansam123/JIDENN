from src.data import Dataset
from src.models import basicFC_model 
from src.models.TransformerModel import TransformerModel
from src.models import BDT_model 
import numpy as np
import tensorflow as tf
import os
import datetime
from src.config.ArgumentParser import ArgumentParser
from src.postprocess.pipeline import postprocess_pipe
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default




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
    
    
    
    num_files = len(datafiles)
    num_dev_files = int(num_files * args.dev_size)
    num_test_files = int(num_files * args.test_size)
    dev_files = datafiles[:num_dev_files]
    test_files = datafiles[num_dev_files:num_dev_files+num_test_files]
    train_files = datafiles[num_dev_files+num_test_files:]

    train, dev, test = [Dataset.get_qg_dataset(files, batch_size=args.batch_size, cut=args.cut, take=args.take, shuffle_buffer=args.shuffle_buffer) for files in [train_files, dev_files, test_files]]
    
    
    if args.normalize and args.model != 'BDT':
        prep_ds = train.take(args.normalization_size).map(lambda x,y,z:x)
        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(prep_ds)
    else:
        normalizer = None

    # creating model
    if args.model == "basic_fc":
        model = basicFC_model.create(args, preprocess=normalizer)
        model.summary()   
    elif args.model == "transformer":
        model = TransformerModel(args,len(jidenn.variables), jidenn.LABELS,  preprocess=normalizer)
        model.summary()   
    elif args.model=='BDT':
        model = BDT_model.create(args)
    else:
        assert False, "Model not implemented"
    
    #callbacks
    callbacks = []
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=args.logdir, histogram_freq=1, embeddings_freq=1)
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
