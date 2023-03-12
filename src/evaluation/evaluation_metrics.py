from typing import List, Dict
import tensorflow as tf


class BinaryRejection(tf.keras.metrics.Metric):
    def __init__(self, name='rejection', label_id=None, threshold=0.5, * args, **kwargs):
        super(BinaryRejection, self).__init__(name=name, *args, **kwargs)
        self.tpr = self.add_weight(name='tp', initializer='zeros')
        self.label_id = label_id
        self.threshold = threshold

    def update_state(self, y_true, y_pred, sample_weight=None):
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
        return 1 - self.tpr

    def reset_states(self):
        self.tpr.assign(0.)

    def get_config(self):
        config = super(BinaryRejection, self).get_config()
        return config


class BinaryEfficiency(tf.keras.metrics.Metric):
    def __init__(self, name='efficiency', label_id=None, threshold=0.5, *args, **kwargs):
        super(BinaryEfficiency, self).__init__(name=name, *args, **kwargs)
        self.tpr = self.add_weight(name='tp', initializer='zeros')
        self.label_id = label_id
        self.threshold = threshold

    def update_state(self, y_true, y_pred, sample_weight=None):
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


class RejectionAtEfficiency(tf.keras.metrics.SpecificityAtSensitivity):
    def __init__(self, name='rejection_at_efficiency', efficiency=0.5, label_id=None, *args, **kwargs):
        super(RejectionAtEfficiency, self).__init__(name=name, sensitivity=efficiency, *args, **kwargs)
        self.label_id = label_id

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        if self.label_id is not None and self.label_id != 1:
            y_true = tf.logical_not(y_true)
            y_pred = 1 - y_pred
        super(RejectionAtEfficiency, self).update_state(y_true, y_pred, sample_weight=sample_weight)

    def get_config(self):
        config = super(RejectionAtEfficiency, self).get_config()
        return config

    def result(self):
        return 1 - super(RejectionAtEfficiency, self).result()


def get_metrics(threshold=0.5) -> List[tf.keras.metrics.Metric]:
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


def calculate_metrics(y_true, score, weights=None, threshold=0.5) -> Dict[str, float]:
    metrics = get_metrics(threshold)
    results = {}
    for metric in metrics:
        metric.reset_states()
        metric.update_state(y_true, score, sample_weight=weights)
        result = metric.result().numpy()
        results[metric.name] = result
    return results
