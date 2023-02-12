from typing import List, Dict
import tensorflow as tf

    
class BinaryRejection(tf.keras.metrics.Metric):
    def __init__(self, name='rejection', label_id=None, *args, **kwargs):
        super(BinaryRejection, self).__init__(name=name,  *args, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.label_id = label_id

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)
        if self.label_id is not None and self.label_id != 1:
            y_true = tf.logical_not(y_true)
            y_pred = tf.logical_not(y_pred)
        
        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_weights(sample_weight, values)
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

        values = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_weights(sample_weight, values)
            values = tf.multiply(values, sample_weight)
        self.false_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        fpr = self.false_positives / (self.false_positives + self.true_positives)
        return 1/fpr

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_positives.assign(0.)
        self.false_positives.assign(0.)

    def get_config(self):
        config = super(BinaryRejection, self).get_config()
        return config

class BinaryEfficiency(tf.keras.metrics.Metric):
    def __init__(self, name='efficiency', label_id=None, *args, **kwargs):
        super(BinaryEfficiency, self).__init__(name=name,  *args, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')
        self.label_id = label_id

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)
        if self.label_id is not None and self.label_id != 1:
            y_true = tf.logical_not(y_true)
            y_pred = tf.logical_not(y_pred)
        
        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_weights(sample_weight, values)
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_weights(sample_weight, values)
            values = tf.multiply(values, sample_weight)
        self.false_negatives.assign_add(tf.reduce_sum(values))

    def result(self):
        tpr = self.true_positives / (self.true_positives + self.false_negatives)
        return tpr

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_positives.assign(0.)
        self.false_negatives.assign(0.)

    def get_config(self):
        config = super(BinaryEfficiency, self).get_config()
        return config
    
class RejectionAtEfficiency(tf.keras.metrics.SpecificityAtSensitivity):
    def __init__(self, name='rejection_at_efficiency', efficiency=0.5, label_id=None, *args, **kwargs):
        super(RejectionAtEfficiency, self).__init__(name=name,  sensitivity=efficiency, *args, **kwargs)
        self.label_id = label_id

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)
        if self.label_id is not None and self.label_id != 1:
            y_true = tf.logical_not(y_true)
            y_pred = tf.logical_not(y_pred)
        super(RejectionAtEfficiency, self).update_state(y_true, y_pred, sample_weight=sample_weight)

    def get_config(self):
        config = super(RejectionAtEfficiency, self).get_config()
        return config
    
    def result(self):
        return 1/(1 - super(RejectionAtEfficiency, self).result())


def get_metrics() -> List[tf.keras.metrics.Metric]:
    metrics = [
        tf.keras.metrics.BinaryCrossentropy(name='loss'),
        tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
        BinaryEfficiency(name='gluon_efficiency', label_id=0),
        BinaryEfficiency(name='quark_efficiency', label_id=1),
        BinaryRejection(name='gluon_rejection', label_id=0),
        BinaryRejection(name='quark_rejection', label_id=1),
        RejectionAtEfficiency(name='gluon_rejection_at_efficiency_0.5', label_id=0),
        RejectionAtEfficiency(name='quark_rejection_at_efficiency_0.5', label_id=1),
        tf.keras.metrics.AUC(name='auc')]
    return metrics


def calculate_metrics(y_true, y_pred, weights=None) -> Dict[str, float]:
    metrics = get_metrics()
    results = {}
    for metric in metrics:
        metric.update_state(y_true, y_pred, sample_weight=weights)
        result = metric.result().numpy()
        results[metric.name] = result
    return results
