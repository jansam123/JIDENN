from logging import Logger
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
#
from .ValidationFigures import ValidationROC, ValidationCM, ValidationScoreHistogram, ValidationLabelHistogram


def plot_validation_figs(df: pd.DataFrame, logdir: str, log: Logger, formats=['png', 'pdf']):
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
    plt.close('all')

def plot_metrics_per_cut(df: pd.DataFrame, logdir: str, log: Logger, formats=['png', 'pdf']):
    base_path = os.path.join(logdir, "metrics")
    os.makedirs(base_path, exist_ok=True)
    format_path = []
    for format in formats:
        format_path.append(os.path.join(base_path, format))
        os.makedirs(format_path[-1], exist_ok=True)

    for metric in df.columns:
        if metric == 'cut':
            continue
        log.info(f"Plotting {metric} for cuts")
        sns.pointplot(x='cut', y=metric, data=df, join=False)
        plt.xlabel('Cut')
        plt.ylabel(str(metric))
        for fmt, path in zip(formats, format_path):
            plt.savefig(os.path.join(path, f'{metric}.{fmt}'))
        plt.close()