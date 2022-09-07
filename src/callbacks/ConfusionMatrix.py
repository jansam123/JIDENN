import matplotlib.pyplot as plt
from  sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf
import itertools
from io import BytesIO




def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 1000, decimals=0).astype(int)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


#  "stolen" from ttps://towardsdatascience.com/exploring-confusion-matrix-evolution-on-tensorboard-e66b39f4ac12
def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """

    buf = BytesIO()

    # Use plt.savefig to save the plot to a PNG in memory.
    plt.savefig(buf, format='png')

    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)

    # Use tf.image.decode_png to convert the PNG buffer
    # to a TF image. Make sure you use 4 channels.
    image = tf.image.decode_png(buf.getvalue(), channels=4)

    # Use tf.expand_dims to add the batch dimension
    image = tf.expand_dims(image, 0)

    return image


class ConfusionMatrix(tf.keras.callbacks.Callback):
    def __init__(self, args, data:tf.data.Dataset, class_names:List[str], labels:list):
        super().__init__()
        self.args = args
        self._data = data
        self._class_names = class_names
        self._labels = labels

    def on_epoch_end(self, epoch, logs):
        if epoch != 0:
            return 
        file_writer_cm = tf.summary.create_file_writer(self.args.logdir + '/cm')
        
        # Use the model to predict the values from the test_images.
        test_pred_raw = self.model.predict(self._data, batch_size=self.args.batch_size)

        test_pred = np.argmax(test_pred_raw, axis=1)

        # Calculate the confusion matrix using sklearn.metrics
        cm = confusion_matrix(self._labels, test_pred)

        figure = plot_confusion_matrix(cm, class_names=self._class_names)
        cm_image = plot_to_image(figure)

        # Log the confusion matrix as an image summary.
        with file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)


