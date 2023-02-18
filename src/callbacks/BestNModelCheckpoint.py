from tensorflow.python.keras.utils import tf_utils
import tensorflow as tf
import shutil
import os
import logging


log = logging.getLogger(__name__)

# Original source: https://github.com/schustmi/tf_utils/blob/915fe5e231ca302b28cd02dc8ac2e4c772a62e0b/tf_utils/callbacks.py#L34


class BestNModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    """Callback to only save the best N checkpoints of a keras model or its weights.
    Arguments:
        filepath: string or `PathLike`, path where the checkpoints will be saved. Make sure
          to pass a format string as otherwise the checkpoint will be overridden each step.
          (see `keras.callbacks.ModelCheckpoint` for detailed formatting options)
        max_to_keep: int, how many checkpoints to keep.
        keep_most_recent: if True, the most recent checkpoint will be saved in addition to
          the best `max_to_keep` ones.
        monitor: name of the metric to monitor. The value for this metric will be used to
          decide which checkpoints will be kept. See `keras.callbacks.ModelCheckpoint` for
          more information.
        mode: one of {'min', 'max'}. Depending on `mode`, the checkpoints with the highest ('max')
          or lowest ('min') values in the monitored quantity will be kept.
        **kwargs: Additional arguments that get passed to `keras.callbacks.ModelCheckpoint`.
    """

    def __init__(self,
                 filepath,
                 max_to_keep: int,
                 keep_most_recent=True,
                 monitor='val_loss',
                 mode='min',
                 **kwargs):
        if kwargs.pop('save_best_only', None):
            log.warning("Setting `save_best_only` to False.")

        if max_to_keep < 1:
            log.warning("BestNModelCheckpoint parameter `max_to_keep` must be greater than 0, setting it to 1.")
            max_to_keep = 1

        super().__init__(filepath,
                         monitor=monitor,
                         mode=mode,
                         save_best_only=False,
                         **kwargs)
        self._keep_count = max_to_keep
        self._checkpoints = {}

        self._keep_most_recent = keep_most_recent
        if self._keep_most_recent:
            self._most_recent_checkpoint = None

    def _save_model(self, epoch, batch, logs):
        super()._save_model(epoch, batch, logs)
        logs = tf_utils.sync_to_numpy_or_python_type(logs or {})
        filepath = self._get_file_path(epoch, batch, logs)

        if not self._checkpoint_exists(filepath):
            # Did not save a checkpoint for current epoch
            return

        value = logs.get(self.monitor)

        if self._keep_most_recent:
            # delay adding to list of current checkpoints until next save
            # if we should always keep the most recent checkpoint
            if self._most_recent_checkpoint:
                self._checkpoints.update(self._most_recent_checkpoint)
            self._most_recent_checkpoint = {filepath: value}
        else:
            self._checkpoints[filepath] = value

        if len(self._checkpoints) > self._keep_count:
            self._delete_worst_checkpoint()

    def _delete_worst_checkpoint(self):
        worst_checkpoint = None  # tuple (filepath, value)

        for checkpoint in self._checkpoints.items():
            if worst_checkpoint is None or self.monitor_op(worst_checkpoint[1], checkpoint[1]):
                worst_checkpoint = checkpoint

        self._checkpoints.pop(worst_checkpoint[0])
        self._delete_checkpoint_files(worst_checkpoint[0])

    @staticmethod
    def _delete_checkpoint_files(checkpoint_path):
        log.info(f"Removing files for checkpoint '{checkpoint_path}'")

        if os.path.isdir(checkpoint_path):
            # SavedModel format delete the whole directory
            shutil.rmtree(checkpoint_path)
            return

        for f in tf.io.gfile.glob(checkpoint_path + '*'):
            os.remove(f)
