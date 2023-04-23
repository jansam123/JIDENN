"""
Module containing custom metrics for evaluating models mainly used in HEP applications. 
"""
from typing import List, Dict, Literal, Optional, Union
import tensorflow as tf
import numpy as np


class BinaryEfficiency(tf.keras.metrics.Metric):
    r"""Binary Efficiency metric.
    It is defined as
    $$\varepsilon_i=\frac{T_i}{T_i+F_i}$$
    where $T_i$ is the number of correctly classified data of class $i$ 
    and $F_i$ is the number of incorrectly classified data of class $i$.

    If $i = 1$, then it is the efficiency is called true positive rate (TPR).

    Args:
        label_id (int, optional): The label id for which the efficiency is calculated. Defaults to 1.
        threshold (float, optional): The threshold for the prediction. Defaults to 0.5.
        name (str, optional): The name of the metric. Defaults to 'efficiency'.
    """

    def __init__(self, label_id: Literal[0, 1] = 1, threshold=0.5, name='efficiency'):
        super(BinaryEfficiency, self).__init__(name=name)
        self.tpr = self.add_weight(name='tp', initializer='zeros')
        self.label_id = label_id
        self.threshold = threshold

    def update_state(self, y_true: Union[tf.Tensor, np.ndarray], y_pred: Union[tf.Tensor, np.ndarray], sample_weight: Union[tf.Tensor, np.ndarray] = None):
        """Accumulates the efficiency.

        Args:
            y_true (tf.Tensor): The true labels.
            y_pred (tf.Tensor): The predicted labels.
            sample_weight (tf.Tensor, optional): The sample weights. Defaults to None.
        """
        y_pred = tf.cast(y_pred > self.threshold, tf.bool)
        y_true = tf.cast(y_true, tf.bool)
        if self.label_id is not None and self.label_id != 1:
            y_true = tf.logical_not(y_true)
            y_pred = tf.logical_not(y_pred)

        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_weights(sample_weight, values)
            values = tf.multiply(values, sample_weight)
        self.tpr.assign_add(tf.reduce_sum(values) / tf.reduce_sum(tf.cast(y_true, self.dtype)))

    def result(self):
        return self.tpr

    def reset_states(self):
        self.tpr.assign(0.)

    def get_config(self):
        config = super(BinaryEfficiency, self).get_config()
        return config


class BinaryRejection(tf.keras.metrics.Metric):
    r"""Binary Rejection metric.
    It is defined as
    $$\varepsilon_i^{-1}=\frac{T_i+F_i}{T_i}$$
    where $T_i$ is the number of correctly classified data of class $i$ 
    and $F_i$ is the number of incorrectly classified data of class $i$.

    Args:
        label_id (int, optional): The label id for which the efficiency is calculated. Defaults to 1.
        threshold (float, optional): The threshold for the prediction. Defaults to 0.5.
        name (str, optional): The name of the metric. Defaults to 'rejection'.
    """

    def __init__(self, label_id: Literal[0, 1] = 1, threshold=0.5, name='rejection'):
        super(BinaryRejection, self).__init__(name=name)
        self.tpr = self.add_weight(name='tp', initializer='zeros')
        self.label_id = label_id
        self.threshold = threshold

    def update_state(self, y_true: Union[tf.Tensor, np.ndarray], y_pred: Union[tf.Tensor, np.ndarray], sample_weight: Union[tf.Tensor, np.ndarray] = None):
        """Accumulates the efficiency.

        Args:
            y_true (tf.Tensor): The true labels.
            y_pred (tf.Tensor): The predicted labels.
            sample_weight (tf.Tensor, optional): The sample weights. Defaults to None.
        """
        y_pred = tf.cast(y_pred > self.threshold, tf.bool)
        y_true = tf.cast(y_true, tf.bool)
        if self.label_id is not None and self.label_id != 1:
            y_true = tf.logical_not(y_true)
            y_pred = tf.logical_not(y_pred)

        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_weights(sample_weight, values)
            values = tf.multiply(values, sample_weight)
        self.tpr.assign_add(tf.reduce_sum(values) / tf.reduce_sum(tf.cast(y_true, self.dtype)))

    def result(self):
        return 1 / self.tpr

    def reset_states(self):
        self.tpr.assign(0.)

    def get_config(self):
        config = super(BinaryRejection, self).get_config()
        return config


class RejectionAtEfficiency(tf.keras.metrics.SpecificityAtSensitivity):
    """Rejection at efficiency metric.
    in this case the threshold is chosen such that the efficiency of one class is equal to the given `efficiency`.

    Args:
        efficiency (float, optional): The efficiency. Defaults to 0.5.
        label_id (int, optional): The label id for which the efficiency is calculated. Defaults to 1.
        name (str, optional): The name of the metric. Defaults to 'rejection_at_efficiency'.
    """

    def __init__(self, efficiency: float = 0.5, label_id: Literal[0, 1] = 1, name='rejection_at_efficiency'):
        super(RejectionAtEfficiency, self).__init__(name=name, sensitivity=efficiency)
        self.label_id = label_id

    def update_state(self, y_true: Union[tf.Tensor, np.ndarray], y_pred: Union[tf.Tensor, np.ndarray], sample_weight: Union[tf.Tensor, np.ndarray] = None):
        """Accumulates the efficiency.

        Args:
            y_true (tf.Tensor): The true labels.
            y_pred (tf.Tensor): The predicted labels.
            sample_weight (tf.Tensor, optional): The sample weights. Defaults to None.
        """
        y_true = tf.cast(y_true, tf.bool)
        if self.label_id is not None and self.label_id != 1:
            y_true = tf.logical_not(y_true)
            y_pred = 1 - y_pred
        super(RejectionAtEfficiency, self).update_state(y_true, y_pred, sample_weight=sample_weight)

    def get_config(self):
        config = super(RejectionAtEfficiency, self).get_config()
        return config

    def result(self):
        return 1 / super(RejectionAtEfficiency, self).result()


def get_metrics(threshold: float = 0.5) -> List[tf.keras.metrics.Metric]:
    """Returns a list of metrics.

    Args:
        threshold (float, optional): The threshold for the prediction. Defaults to 0.5.

    Returns:
        List[tf.keras.metrics.Metric]: The list of selected metrics.
    """
    metrics = [
        tf.keras.metrics.BinaryCrossentropy(name='loss'),
        tf.keras.metrics.BinaryAccuracy(name='binary_accuracy', threshold=threshold),
        BinaryEfficiency(name='gluon_efficiency', label_id=0, threshold=threshold),
        BinaryEfficiency(name='quark_efficiency', label_id=1, threshold=threshold),
        BinaryRejection(name='gluon_rejection', label_id=0, threshold=threshold),
        BinaryRejection(name='quark_rejection', label_id=1, threshold=threshold),
        RejectionAtEfficiency(name='gluon_rej_at_quark_eff_0.9', label_id=0, efficiency=0.9),
        RejectionAtEfficiency(name='quark_rej_at_gluon_eff_0.9', label_id=1, efficiency=0.9),
        tf.keras.metrics.AUC(name='auc')]
    return metrics


def calculate_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float = 0.5, weights: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Calculates the metrics.
    Args:
        y_true (np.ndarray): The true labels.
        score (np.ndarray): The output scores of the model.
        threshold (float, optional): The threshold for the prediction. Defaults to 0.5.
        weights (np.ndarray, optional): The sample weights. Defaults to None.
    Returns:
        Dict[str, float]: The dictionary of metric names and values.
    """
    metrics = get_metrics(threshold)
    results = {}
    for metric in metrics:
        metric.reset_states()
        metric.update_state(y_true, score, sample_weight=weights)
        result = metric.result().numpy()
        results[metric.name] = result
    return results
