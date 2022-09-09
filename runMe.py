import numpy as np
import tensorflow as tf
import os
from datetime import datetime
import logging
import hydra
from hydra.core.config_store import ConfigStore
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

from src.data import Dataset
from src.models import BDT, basicFC, transformer
from src.postprocess.pipeline import postprocess_pipe
from src.config import config

cs = ConfigStore.instance()
cs.store(name="args", node=config.JIDENNConfig)


@hydra.main(version_base="1.2",config_path="src/config", config_name="config")
def main(args: config.JIDENNConfig) -> None:
    log = logging.getLogger(__name__)
    
    #debug mode for tensorflow
    if args.params.debug:
        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()

    #fixing seed for reproducibility
    np.random.seed(args.params.seed)
    tf.random.set_seed(args.params.seed)
    
    # managing threads
    tf.config.threading.set_inter_op_parallelism_threads(args.params.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.params.threads)
    
    
    #dataset preparation
    datafiles = [os.path.join(args.params.data_path, folder, file+f':{args.data.tttree_name}') for folder in os.listdir(args.params.data_path) for file in os.listdir(os.path.join(args.params.data_path, folder)) if '.root' in file]
    np.random.shuffle(datafiles)
    
    num_files = len(datafiles)
    
    dev_size = int(args.dataset.take*args.dataset.dev_size) if args.dataset.take is not None else None
    num_dev_files = int(num_files * args.dataset.dev_size)
    dev_files = datafiles[:num_dev_files]
    
    test_size = int(args.dataset.take*args.dataset.test_size) if args.dataset.take is not None else None
    num_test_files = int(num_files * args.dataset.test_size)
    test_files = datafiles[num_dev_files:num_dev_files+num_test_files]
    
    train_files = datafiles[num_dev_files+num_test_files:]
    
    sizes = [args.dataset.take, dev_size, test_size]
    dataset_files = [train_files, dev_files, test_files]
    
    if args.params.model == 'BDT':
        args.data.cut = f'({args.data.cut})&({args.data.weight}>0)' if args.data.cut is not None else f"{args.data.weight}>0"

    train, dev, test = [Dataset.get_qg_dataset(files, args_data=args.data, args_dataset=args.dataset, size=size) for files, size in zip(dataset_files, sizes)]
    
    
    if args.preprocess.normalize and args.params.model != 'BDT':
        prep_ds = train.take(args.preprocess.normalization_size) if args.preprocess.normalization_size is not None else train
        prep_ds=prep_ds.map(lambda x,y,z:x)
        normalizer = tf.keras.layers.Normalization(axis=-1)
        log.info("Getting std and mean of the dataset...")
        log.info(f"Subsample size: {args.preprocess.normalization_size}")
        normalizer.adapt(prep_ds)
    else:
        normalizer = None

    # creating model
    if args.params.model == "basic_fc":
        model = basicFC.create(args.params, args.basic_fc, args.data, preprocess=normalizer)
        model.summary(print_fn=log.info)   
    elif args.params.model == "transformer":
        model = transformer.create(args.params, args.transformer, args.data, preprocess=normalizer)
        model.summary(print_fn=log.info)   
    elif args.params.model=='BDT':
        model = BDT.create(args.bdt)
    else:
        assert False, "Model not implemented"
    
    #callbacks
    callbacks = []
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=args.params.logdir, histogram_freq=1, embeddings_freq=1)
    callbacks += [tb_callback]
    
    class LogCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            self.__epoch_start = None
        
        def on_epoch_begin(self, epoch, logs=None):
            self.__epoch_start = datetime.utcnow()
            
        def on_epoch_end(self, epoch, logs):
            eta = datetime.utcnow() - self.__epoch_start if self.__epoch_start is not None else '-:--:--'
            eta = str(eta).split('.')[0]
            log_str = f"Epoch {epoch+1}/{args.params.epochs}: "
            log_str += f"ETA: {eta} - "
            log_str +=  " - ".join([f"{k}: {v:.4}" for k,v in logs.items()])
            log.info(log_str)
            
    callbacks += [LogCallback()]
    
    #running training
    model.fit(train, validation_data=dev, epochs=args.params.epochs, callbacks=callbacks, validation_steps=args.dataset.validation_batches if args.dataset.take is None else None)
        
    
    test_dataset = test.unbatch().map(lambda d, l, w: d).batch(args.dataset.batch_size)
    test_dataset_labels = test.unbatch().map(lambda x,y,z: y)
    
    postprocess_pipe(model, test_dataset,test_dataset_labels, args.params.logdir, log)

if __name__ == "__main__":
    main()
