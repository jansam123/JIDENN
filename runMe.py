import numpy as np
import tensorflow as tf
import os
import logging
import hydra
from hydra.core.config_store import ConfigStore
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

from src.data import Dataset
from src.models import basicFC, transformer #, BDT
from src.postprocess.pipeline import postprocess_pipe
from src.config import config
from src.callbacks.get_callbacks import get_callbacks

cs = ConfigStore.instance()
cs.store(name="args", node=config.JIDENNConfig)


@hydra.main(version_base="1.2",config_path="src/config", config_name="config")
def main(args: config.JIDENNConfig) -> None:
    log = logging.getLogger(__name__)
    args.data.input_size = len(args.data.variables.perJet) if args.data.variables.perJet is not None else 0
    args.data.input_size += len(args.data.variables.perEvent) if args.data.variables.perEvent is not None else 0
    args.data.input_size += len(args.data.variables.perJetTuple) if args.data.variables.perJetTuple is not None else 0
    
    # GPU logging
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) == 0:
        log.warning("No GPU found, using CPU")
    for i, gpu in enumerate(gpus):
        gpu_info = tf.config.experimental.get_device_details(gpu)
        log.info(f"GPU {i}: {gpu_info['device_name']} with compute capability {gpu_info['compute_capability'][0]}.{gpu_info['compute_capability'][1]}")


    
    #debug mode for tensorflow
    if args.params.debug:
        log.info("Debug mode enabled")
        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()

    #fixing seed for reproducibility
    if args.params.seed is not None:
        log.info(f"Setting seed to {args.params.seed}")
        np.random.seed(args.params.seed)
        tf.random.set_seed(args.params.seed)
    
    # managing threads
    tf.config.threading.set_inter_op_parallelism_threads(args.params.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.params.threads)
    
    mirrored_strategy = tf.distribute.MirroredStrategy() if len(gpus) > 1 else None
    
    
    #dataset preparation
    datafiles = [os.path.join(args.data.path, folder, file+f':{args.data.tttree_name}') for folder in os.listdir(args.data.path) for file in os.listdir(os.path.join(args.data.path, folder)) if '.root' in file]
    np.random.shuffle(datafiles)
    
    if len(datafiles) == 0:
        log.error("No data found!")
        raise FileNotFoundError("No data found!")
    
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

    train = Dataset.get_qg_dataset(train_files, args_data=args.data, args_dataset=args.dataset, size=args.dataset.take, name="train")
    
    if num_dev_files == 0:
        dev = None
        log.warning("No dev dataset, skipping validation")
    else:
        dev = Dataset.get_qg_dataset(dev_files, args_data=args.data, args_dataset=args.dataset, size=dev_size, name="dev")
        
    if num_test_files == 0:
        test = None
        log.warning("No test dataset, skipping evaluation")
    else:
        test = Dataset.get_qg_dataset(test_files, args_data=args.data, args_dataset=args.dataset, size=test_size, name="test")

    def _model():
        if args.preprocess.normalize and args.params.model != 'BDT':
            prep_ds = train.take(args.preprocess.normalization_size) if args.preprocess.normalization_size is not None else train
            prep_ds=prep_ds.map(lambda x,y,z: x[0]) if args.data.variables.perJetTuple is not None else prep_ds.map(lambda x1,y1,z1: x1)
            normalizer = tf.keras.layers.Normalization(axis=-1)
            log.info("Getting std and mean of the dataset...")
            log.info(f"Subsample size: {args.preprocess.normalization_size}")
            normalizer.adapt(prep_ds)
        else:
            normalizer = None
            
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
            
        return model
    
    # creating model
    if len(gpus) < 2:
        model = _model()
    else:
        # mirrored_strategy = tf.distribute.MirroredStrategy()
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            model = _model()
        
    #callbacks
    callbacks = get_callbacks(args.params, log)
    
    #running training
    model.fit(train, validation_data=dev, epochs=args.params.epochs, callbacks=callbacks, validation_steps=args.dataset.validation_batches if args.dataset.take is None else None)
        
    if test is None:
        log.warning("No test dataset, skipping evaluation.")
        return 
    
    #split train into labels and features
    test_dataset = test.unbatch().map(lambda d, l, w: d).batch(args.dataset.batch_size)
    test_dataset_labels = test.unbatch().map(lambda x,y,z: y)
    postprocess_pipe(model, test_dataset,test_dataset_labels, args.params.logdir, log)

if __name__ == "__main__":
    main()
