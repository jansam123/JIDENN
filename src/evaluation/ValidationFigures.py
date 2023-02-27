from abc import abstractmethod
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix, auc
from io import BytesIO
from typing import List, Union

sns.set_theme(style="dark")


class ValidationFigure:
    def __init__(self, df: pd.DataFrame, name: str = 'fig', class_names: Union[List[str], None] = None):
        self._df = df
        self._name = name
        self._class_names = class_names
        self._fig = self._get_fig()

    @property
    def figure(self):
        return self._fig

    @property
    def figure_name(self):
        return self._name

    @abstractmethod
    def _get_fig(self, fig: Union[plt.Figure, None] = None) -> plt.Figure:
        "Method that creates matplotlib figure."

    def save_fig(self, path: str, format: str = 'png'):
        self._fig.savefig(os.path.join(path, self._name + f".{format}"), dpi=300)

    def to_tensorboard(self, path: str):
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
    def _get_fig(self, fig: Union[plt.Figure, None] = None) -> plt.Figure:
        fp, tp, _ = roc_curve(self._df['label'].values, self._df['score'].values)
        auc_score = auc(fp, tp)

        if fig is None:
            fig = plt.figure(figsize=(8, 8))
        sns.lineplot(x=100*fp, y=100*tp, label=f'AUC = {auc_score:.3f}', linewidth=2)
        sns.lineplot(x=[0, 50, 100], y=[0, 50, 100], label=f'Random',
                     linewidth=1, linestyle='--', color='darkred', alpha=0.5)
        plt.plot([0, 0, 100], [0, 100, 100], color='darkgreen', linestyle='-.', label='Ideal', alpha=0.5)
        plt.xlabel('False positives [%]')
        plt.ylabel('True positives [%]')
        plt.grid(True)
        plt.legend()
        ax = plt.gca()
        ax.set_aspect('equal')
        return fig


class ValidationCM(ValidationFigure):

    def _get_fig(self, fig: Union[plt.Figure, None] = None) -> plt.Figure:
        cm = confusion_matrix(self._df['label'].values, self._df['prediction'].values)
        if fig is None:
            fig = plt.figure(figsize=(6, 6))

        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 1000, decimals=0).astype(int)
        df_cm = pd.DataFrame(cm, index=self._class_names, columns=self._class_names)
        sns.heatmap(df_cm, annot=True, fmt='4d', cmap=plt.cm.Blues, cbar=False)
        plt.title("Confusion matrix")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return fig


class ValidationScoreHistogram(ValidationFigure):
    def _get_fig(self, fig: Union[plt.Figure, None] = None) -> plt.Figure:
        if fig is None:
            fig = plt.figure(figsize=(8, 8))
        sns.kdeplot(data=self._df, x='score', hue='named_label', fill=True, palette='Set1', alpha=0.1, linewidth=2.5)
        plt.xlabel('Score')
        return fig


class ValidationLabelHistogram(ValidationFigure):
    def _get_fig(self, fig: Union[plt.Figure, None] = None) -> plt.Figure:
        if fig is None:
            fig = plt.figure(figsize=(8, 8))
        sns.histplot(self._df, x='named_prediction', hue='named_label', stat='count', multiple='stack')
        plt.xlabel('Predicted Tag')
        return fig
