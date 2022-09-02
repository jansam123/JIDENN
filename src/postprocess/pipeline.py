from src.config.ArgumentParser import ArgumentParser
import tensorflow as tf
import numpy as np
from .confusion_matrix import get_cm
from .roc_curve import get_roc
from .prediction_hist import get_pred_hist, get_pred_violin
import os


    

def postprocess_pipe(model, data:tf.data.Dataset, labels:tf.data.Dataset, args:ArgumentParser):
    np_labels = np.array(list(labels.as_numpy_iterator()), dtype=np.int32)
    
    res = model.predict(data, batch_size=args.batch_size)
    pred = res.round()[:,0]
    print(f"Accuracy: {np.mean(pred == np_labels):.4f}")
    
    with open(os.path.join(args.logdir, "predictions.txt"), "w") as f:
        for i,j,k in zip(data.unbatch(), res, np_labels):
            print(f'data={i} \n prediction={j} \n label={k} \n', file=f)
        
    base_path = os.path.join(args.logdir, "figs")
    os.makedirs(base_path, exist_ok=True)
    get_cm(np_labels, pred, labels=['gluon', 'quark'], path=os.path.join(base_path, "confusion_matrix.png"))
    get_roc(np_labels, pred, path=os.path.join(base_path, "roc_curve.png"))
    get_pred_hist(res, path=os.path.join(base_path, "prediction_hist.png"))
    get_pred_hist(np_labels, path=os.path.join(base_path, "label_hist.png"))
    get_pred_violin([np_labels,pred, res[:,0]], path=os.path.join(base_path, "prediction_violin.png"))
    
    