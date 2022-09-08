import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from  sklearn.metrics import roc_curve,confusion_matrix, auc
from itertools import product
from io import BytesIO



class ValFigures:
        
    def __init__(self, predictions:np.ndarray, labels:np.ndarray, argmax_predictions:np.ndarray, class_names:list[str]):
        self._class_names = class_names
        self._predictions = predictions
        self._labels = labels
        self._argmax_predictions = argmax_predictions
        
        self._figures, self._figure_names = self._get_figures()        
        
    @property
    def figures(self):
        return self._figures
    
    @property
    def figure_names(self):
        return self._figure_names

    def to_tensorboard(self, path:str):        
        for fig, name in zip(self._figures, self._figure_names):
            with tf.summary.create_file_writer(path).as_default():
                tf.summary.image(name, self._plot_to_image(fig), step=0)
        
    def _plot_to_image(self, figure):
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
        
    def save_figures(self, path:str, format:str='png'):
        for fig, name in zip(self._figures, self._figure_names):
            fig.savefig(os.path.join(path, name + f".{format}"))
        
    def _get_figures(self):
        figures = []
        figures.append(self._confusion_matrix(self._labels, self._argmax_predictions))
        figures.append(self._roc(self._labels, self._predictions))
        figures.append(self._violin([self._labels, self._predictions, self._argmax_predictions], x_labels=['labels', 'predictions', 'argmax_predictions']))
        figures.append(self._hist(self._predictions))
        figures.append(self._hist(self._labels))
        figure_names = ['confusion_matrix', 'roc', 'violin', 'predictions', 'labels']
        return figures, figure_names
        
    
    def _hist(self, predictions):
        fig = plt.figure(figsize=(8, 8))
        plt.hist(predictions, label='predictions', alpha=0.5)
        plt.xlabel('Prediction probability')
        plt.ylabel('Count')
        return fig
    
    def _violin(self, predictions:list[np.ndarray] = [], x_labels: list[str] | None = None):
        fig = plt.figure(figsize=(8, 8))
        sns.violinplot(data=predictions)
        if x_labels is not None:
            plt.xticks(range(len(predictions)), x_labels)
        plt.xlabel('Prediction probability')
        plt.ylabel('Count')
        return fig
    
    def _roc(self, labels, predictions):
        fp, tp, _ = roc_curve(labels, predictions)
        auc_score = auc(fp, tp)
        
        fig = plt.figure(figsize=(8, 8))
        plt.plot(100*fp, 100*tp, label=f'AUC = {auc_score:.3f}', linewidth=2)
        plt.xlabel('False positives [%]')
        plt.ylabel('True positives [%]')
        plt.grid(True)
        plt.legend()
        ax = plt.gca()
        ax.set_aspect('equal')
        return fig
    
    def _confusion_matrix(self, labels, predictions):
        cm = confusion_matrix(labels, predictions)
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)  # type: ignore
        plt.title("Confusion matrix")
        tick_marks = np.arange(len(self._class_names))
        plt.xticks(tick_marks, self._class_names, rotation=45)
        plt.yticks(tick_marks, self._class_names)

        # Normalize the confusion matrix.
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 1000, decimals=0).astype(int)

        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.

        for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return fig