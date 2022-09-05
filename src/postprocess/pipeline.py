from src.config.ArgumentParser import ArgumentParser
import tensorflow as tf
import numpy as np
import os
from .ValFigures import ValFigures

    
    
def postprocess_pipe(model, data:tf.data.Dataset, labels:tf.data.Dataset, args:ArgumentParser):
    np_labels = np.array(list(labels.as_numpy_iterator()), dtype=np.int32)
    
    res = model.predict(data, batch_size=args.batch_size)
    pred = res.round()[:,0]
    res = res[:,0]
    print(f"Accuracy: {np.mean(pred == np_labels):.4f}")
    
    with open(os.path.join(args.logdir, "predictions.txt"), "w") as f:
        for i,j,k in zip(data.unbatch(), res, np_labels):
            print(f'data={i} \n prediction={j} \n label={k} \n', file=f)
        
    base_path = os.path.join(args.logdir, "figs")
    tb_base_path = os.path.join(args.logdir, "plots")
    os.makedirs(base_path, exist_ok=True)
    
    print("Generating validation figures...")
    val_figs = ValFigures(res, np_labels, pred, ['gluon', 'quark'])
    
    print("Saving to tensorboard...")
    val_figs.to_tensorboard(tb_base_path)
    
    print("Saving figures to disk...")
    for format in ['png', 'pdf']:
        format_path = os.path.join(base_path, format)
        os.makedirs(format_path, exist_ok=True)
        val_figs.save_figures(os.path.join(base_path, format), format=format)