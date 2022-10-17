from logging import Logger
import tensorflow as tf
import numpy as np
import os
from .ValidationFigures import ValidationFigures

    
    
def postprocess_pipe(model, data:tf.data.Dataset, labels:tf.data.Dataset, logdir:str, log:Logger):
    np_labels = np.array(list(labels.as_numpy_iterator()), dtype=np.int32)
    

    res = model.predict(data).ravel()
    pred = res.round()
    
    log.info(f"Test accuracy: {np.mean(pred == np_labels):.4f}")
    
    with open(os.path.join(logdir, "predictions.txt"), "w") as f:
        for i,j,k in zip(data.unbatch().take(100), res, np_labels):
            print(f'data={i} \n prediction={j} \n label={k} \n', file=f)
        
    base_path = os.path.join(logdir, "figs")
    tb_base_path = os.path.join(logdir, "plots")
    os.makedirs(base_path, exist_ok=True)
    
    log.info("Generating validation figures...")
    val_figs = ValidationFigures(res, np_labels, pred, ['gluon', 'quark'])
    
    log.info("Saving figures to disk...")
    for format in ['png', 'pdf']:
        format_path = os.path.join(base_path, format)
        os.makedirs(format_path, exist_ok=True)
        val_figs.save_figures(os.path.join(base_path, format), format=format)
    
    log.info("Saving to tensorboard...")
    val_figs.to_tensorboard(tb_base_path)