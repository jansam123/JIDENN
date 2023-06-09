"""
Module for plotting the results of the evaluation and plotting the training history.
The ploting is done by subsclassing the `jidenn.evaluation.plotter.ValidationFigure` class,
with the `jidenn.evaluation.plotter.ValidationFigure.get_fig` method being the main method.

Each figure is saved as a png file, as a csv file, and saved to tensorboard.
"""
from logging import Logger
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from abc import abstractmethod
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, auc
from io import BytesIO
from typing import List, Union, Dict, Optional


sns.set_theme(style="ticks")


class ValidationFigure:
    """Base class for validation figures.

    Only the `get_fig` method needs to be implemented.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        name (str, optional): The name of the figure. Defaults to 'fig'.
        class_names (List[str], optional): The names of the classes. Defaults to None.

    """

    def __init__(self, df: pd.DataFrame, name: str = 'fig', class_names: Union[List[str], None] = None):
        self._df = df
        self._name = name
        self._class_names = class_names
        self._data = None
        self._fig = self.get_fig()

    @property
    def figure(self):
        """Returns the matplotlib figure."""
        return self._fig

    @property
    def figure_name(self):
        """Returns the name of the figure."""
        return self._name

    @abstractmethod
    def get_fig(self, fig: Union[plt.Figure, None] = None) -> plt.Figure:
        "Creates matplotlib figure."

    def save_fig(self, path: str, format: str = 'png'):
        """Saves the figure to the specified path."""
        self._fig.savefig(os.path.join(path, self._name +
                          f".{format}"), dpi=300, bbox_inches='tight')

    def save_data(self, path: str):
        """Saves the data to the specified path as a csv file."""
        if self._data is None:
            return
        self._data.to_csv(os.path.join(path, self._name + ".csv"))

    def to_tensorboard(self, path: str):
        """Saves the figure to tensorboard."""
        with tf.summary.create_file_writer(path).as_default():
            tf.summary.image(self._name, self._fig_to_image(self._fig), step=0)

    def _fig_to_image(self, figure):
        """
        Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call.
        """
        buf = BytesIO()

        # Use plt.savefig to save the plot to a PNG in memory.
        figure.savefig(buf, format='png')

        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        # plt.close(figure)
        buf.seek(0)
        # Use tf.image.decode_png to convert the PNG buffer
        # to a TF image. Make sure you use 4 channels.
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Use tf.expand_dims to add the batch dimension
        image = tf.expand_dims(image, 0)
        return image


class ValidationROC(ValidationFigure):
    """Class for plotting the ROC curve.
    The ROC curve is calculated using the `sklearn.metrics.roc_curve` function.
    """

    def get_fig(self, fig: Union[plt.Figure, None] = None) -> plt.Figure:
        fp, tp, th = roc_curve(
            self._df['label'].values, self._df['score'].values)
        self._data = pd.DataFrame({'FPR': fp, 'TPR': tp, 'threshold': th})
        auc_score = auc(fp, tp)

        if fig is None:
            fig = plt.figure(figsize=(8, 8))
        sns.lineplot(x=100 * fp, y=100 * tp,
                     label=f'AUC = {auc_score:.3f}', linewidth=2)
        sns.lineplot(x=[0, 50, 100], y=[0, 50, 100], label=f'Random',
                     linewidth=1, linestyle='--', color='darkred', alpha=0.5)
        plt.plot([0, 0, 100], [0, 100, 100], color='darkgreen',
                 linestyle='-.', label='Ideal', alpha=0.5)
        plt.xlabel('False positives [%]')
        plt.ylabel('True positives [%]')
        plt.grid(True)
        plt.legend()
        ax = plt.gca()
        ax.set_aspect('equal')
        return fig


class ValidationCM(ValidationFigure):
    """Plots the confusion matrix."""

    def get_fig(self, fig: Union[plt.Figure, None] = None) -> plt.Figure:
        cm = confusion_matrix(
            self._df['label'].values, self._df['prediction'].values)
        if fig is None:
            fig = plt.figure(figsize=(6, 6))

        cm = np.around(cm.astype('float') / cm.sum(axis=1)
                       [:, np.newaxis] * 1000, decimals=0).astype(int)
        df_cm = pd.DataFrame(cm, index=self._class_names,
                             columns=self._class_names)
        self._data = df_cm
        sns.heatmap(df_cm, annot=True, fmt='4d', cmap=plt.cm.Blues, cbar=False)
        plt.title("Confusion matrix")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return fig


class ValidationScoreHistogram(ValidationFigure):
    """Plots the output scores of the model, colored by the truth label."""

    def get_fig(self, fig: Union[plt.Figure, None] = None) -> plt.Figure:
        if fig is None:
            fig = plt.figure(figsize=(8, 8))
        self._data = self._df[['score', 'Truth Label']]
        ax = sns.histplot(data=self._df, x='score', hue='Truth Label',
                          palette='Set1', stat='count', element="step", fill=True,
                          hue_order=self._class_names)
        sns.move_legend(ax, 'upper center')
        plt.xlabel('Score')
        return fig


class ValidationLabelHistogram(ValidationFigure):
    """Plots the predicted labels of the model, colored by the truth label.
    This is usefull to see if the model is biased towards one class.
    """

    def get_fig(self, fig: Union[plt.Figure, None] = None) -> plt.Figure:
        if fig is None:
            fig = plt.figure(figsize=(8, 8))
        self._data = self._df[['Truth Label', 'named_prediction']]
        sns.histplot(self._df, x='named_prediction', hue='Truth Label',
                     stat='count', multiple='stack', hue_order=self._class_names,
                     palette='Set1')
        plt.xlabel('Predicted Tag')
        return fig


def plot_validation_figs(df: pd.DataFrame, logdir: str, log: Logger, formats: List[str] = ['jpg', 'pdf'], class_names: Optional[List[str]] = None):
    """Plots the validation figures and saves them to disk.
    Args:
        df (pd.DataFrame): The dataframe containing the truth lables, the model output scores and the predictions.
        logdir (str): The directory where the figures are saved.
        log (Logger): The logger.
        formats (list, optional): The formats in which the figures are saved. Defaults to ['jpg', 'pdf'].

    """
    base_path = os.path.join(logdir, "figs")
    tb_base_path = os.path.join(logdir, "plots")
    csv_path = os.path.join(base_path, 'csv')
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(tb_base_path, exist_ok=True)
    os.makedirs(csv_path, exist_ok=True)
    format_path = []
    for format in formats:
        format_path.append(os.path.join(base_path, format))
        os.makedirs(format_path[-1], exist_ok=True)

    figure_classes = [ValidationROC, ValidationCM,
                      ValidationScoreHistogram, ValidationLabelHistogram]
    figure_names = ['roc', 'confusion_matrix', 'score_hist', 'prediction_hist']

    for validation_fig, name in zip(figure_classes, figure_names):
        log.info(f"Generating figure {name}")
        val_fig = validation_fig(df, name, class_names=class_names)
        for fmt, path in zip(formats, format_path):
            val_fig.save_fig(path, fmt)
        val_fig.save_data(csv_path)
        val_fig.to_tensorboard(tb_base_path)
    plt.close('all')


def plot_metrics_per_cut(df: pd.DataFrame, logdir: str, log: Logger, formats=['png', 'pdf']):
    """Plots the metrics for different cuts and saves them to disk.

    Args:
        df (pd.DataFrame): dataframe containing the metrics for different cuts. 
        logdir (str): The directory where the figures are saved.
        log (Logger): The logger.
        formats (list, optional): The formats in which the figures are saved. Defaults to ['jpg', 'pdf'].
    """
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


def plot_train_history(data: List[float], logdir: str, name: str, epochs: int):
    """Plots the training history and saves it to disk.

    Args:
        data (list): The metric values as a function of the epoch.
        logdir (str): The directory where the figures are saved.
        name (str): The name of the metric.
        epochs (int): The number of epochs.
    """
    fig = plt.figure(figsize=(10, 5))
    g = sns.lineplot(data=data, linewidth=2.5, palette='husl')
    g.set(xlabel='Epoch', ylabel=name)
    g.set_xticks(range(epochs))
    g.set_xticklabels(range(1, epochs + 1))
    plt.grid(True)
    fig.savefig(f'{logdir}/{name}.png')
    plt.close()
