from logging import Logger
import pandas as pd 
import numpy as np
import os
#
from .ValidationFigures import ValidationROC, ValidationCM, ValidationScoreHistogram, ValidationLabelHistogram
    
    
def postprocess_pipe(df:pd.DataFrame, logdir:str, log:Logger, formats=['png', 'pdf']):
    
    log.info(f"Test accuracy: {np.mean(df['prediction'].values == df['label'].values):.4f}")
    
    # with open(os.path.join(logdir, "predictions.txt"), "w") as f:
    #     for i,j,k in zip(data.unbatch().take(100), res, np_labels):
    #         print(f'data={i} \n prediction={j} \n label={k} \n', file=f)
        
    base_path = os.path.join(logdir, "figs")
    tb_base_path = os.path.join(logdir, "plots")
    os.makedirs(base_path, exist_ok=True)
    format_path = []
    for format in formats:
        format_path.append(os.path.join(base_path, format))
        os.makedirs(format_path[-1], exist_ok=True)  
    
    figure_classes = [ValidationROC, ValidationCM, ValidationScoreHistogram, ValidationLabelHistogram]
    figure_names = ['roc', 'confusion_matrix', 'score_hist', 'prediction_hist']
        
    for validation_fig, name in zip(figure_classes, figure_names):
        log.info(f"Generating figure {name}")
        val_fig = validation_fig(df, name, ['gluon', 'quark'])
        for fmt, path in zip(formats, format_path):
            val_fig.save_fig(path, fmt)
        val_fig.to_tensorboard(tb_base_path)