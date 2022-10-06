import numpy as np
import tensorflow as tf
import os
import logging
import hydra
from hydra.core.config_store import ConfigStore
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

from src.data import Dataset
from src.models import basicFC, transformer, BDT
from src.postprocess.pipeline import postprocess_pipe
from src.config import config
from src.callbacks.get_callbacks import get_callbacks
import src.data.data_info as data_info

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
        tf.data.experimental.enable_debug_mode()
        tf.config.run_functions_eagerly(True)

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
    # np.random.shuffle(datafiles)
    
    if len(datafiles) == 0:
        log.error("No data found!")
        raise FileNotFoundError("No data found!")
    
    
    if args.params.model == 'BDT':
        args.data.cut = f'({args.data.cut})&({args.data.weight}>0)' if args.data.cut is not None else f"{args.data.weight}>0"

    #TODO: implement when take=None
    train, dev, test  = Dataset.get_qg_dataset(datafiles, args_data=args.data, args_dataset=args.dataset, name="train")
    
    if args.data.draw_distribution is not None:
        log.info(f"Drawing data distribution with {args.data.draw_distribution} samples")
        data_info.generate_data_distributions([train, dev, test], f'{args.params.logdir}/dist', 
                                                      size=args.data.draw_distribution, 
                                                      var_names=args.data.variables.perJet, 
                                                      datasets_names=["train", "dev", "test"])

 
    def _model():
        if args.preprocess.normalize and args.params.model != 'BDT':
            def norm_preprocess(x, y, z):
                # if args.data.variables.perJetTuple is not None:
                #     return x[0]
                # else:
                return x
                 
            prep_ds = train.take(args.preprocess.normalization_size) if args.preprocess.normalization_size is not None else train
            prep_ds = prep_ds.map(norm_preprocess)
            normalizer = tf.keras.layers.Normalization(axis=-1)
            log.info("Getting std and mean of the dataset...")
            log.info(f"Subsample size: {args.preprocess.normalization_size}")
            normalizer.adapt(prep_ds)
            
            if args.data.draw_distribution is not None:
                log.info(f"Drawing data distribution with {args.data.draw_distribution} samples AFTER NORMALIZATION")
                data_info.generate_data_distributions([train, dev, test], f'{args.params.logdir}/dist_postNorm', 
                                                      size=args.data.draw_distribution, 
                                                      var_names=args.data.variables.perJet, 
                                                      datasets_names=["train", "dev", "test"],
                                                      func=normalizer)
            
            
        else:
            normalizer = None
            
        if args.params.model == "basic_fc":
            # model = basicFC.create(args.params, args.basic_fc, args.data, preprocess=normalizer)
            model = tf.keras.Sequential([
                tf.keras.Input(shape=(args.data.input_size,)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer=tf.keras.optimizers.Adam(),
                          loss=tf.keras.losses.BinaryCrossentropy(),
                          metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                   tf.keras.metrics.AUC(name='auc')])
                
            model.summary(print_fn=log.info)   
            
        elif args.params.model == "transformer":
            model = transformer.create(args.params, args.transformer, args.data, preprocess=normalizer)
            model.summary(print_fn=log.info)   
        
            
        elif args.params.model=='BDT':
            model = BDT.create(args.bdt)
            
        else:
            raise NotImplementedError("Model not implemented")
            
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
    callbacks = get_callbacks(args.params, log, dev)
    
    #running training
    # tf.keras.utils.plot_model(model, f"{args.params.logdir}/model.png", show_shapes=True, expand_nested=True)
    model.fit(train, validation_data=dev, epochs=args.params.epochs, callbacks=callbacks, validation_steps=args.dataset.validation_batches if args.dataset.take is None else None)
        
    if test is None:
        log.warning("No test dataset, skipping evaluation.")
        return 
    
    #split train into labels and features
    # def split(x,y,z):
    #     return (x[0], x[1]) 
    print(model.evaluate(train.unbatch().map(lambda *h: (h[0], h[1])).batch(args.dataset.batch_size)))
    test_dataset = dev.unbatch().map(lambda*y: y[0]).batch(args.dataset.batch_size)
    test_dataset_labels = dev.unbatch().map(lambda *x: x[1])
    postprocess_pipe(model, test_dataset, test_dataset_labels, args.params.logdir, log)

if __name__ == "__main__":
    main()
