"""
Module containing custom metrics for evaluating models mainly used in HEP applications. 
"""
from typing import List, Dict, Literal, Optional, Union, Tuple
import tensorflow as tf
# import tensorflow_probability as tfp
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
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')
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
            self.tp.assign_add(tf.reduce_sum(values*sample_weight))
            self.total.assign_add(tf.reduce_sum(tf.cast(y_true, self.dtype)*sample_weight))
        else:
            self.tp.assign_add(tf.reduce_sum(values))
            self.total.assign_add(tf.reduce_sum(tf.cast(y_true, self.dtype)))
            

    def result(self):
        return self.tp / self.total

    def reset_state(self):
        self.tp.assign(0.)
        self.total.assign(0.)

    def get_config(self):
        config = super(BinaryEfficiency, self).get_config()
        config.update({'threshold': self.threshold, 'label_id': self.label_id})
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
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')
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
            self.tp.assign_add(tf.reduce_sum(values*sample_weight))
            self.total.assign_add(tf.reduce_sum(tf.cast(y_true, self.dtype)*sample_weight))
        else:
            self.tp.assign_add(tf.reduce_sum(values))
            self.total.assign_add(tf.reduce_sum(tf.cast(y_true, self.dtype)))

    def result(self):
        tpr = self.tp / self.total
        return 1.0 / (1 - tpr)

    def reset_state(self):
        self.tp.assign(0.)
        self.total.assign(0.)

    def get_config(self):
        config = super(BinaryRejection, self).get_config()
        config.update({'threshold': self.threshold, 'label_id': self.label_id})
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
        super(RejectionAtEfficiency, self).__init__(
            name=name, sensitivity=efficiency)
        self.label_id = label_id
        self.efficiency = efficiency

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
        super(RejectionAtEfficiency, self).update_state(
            y_true, y_pred, sample_weight=sample_weight)

    def get_config(self):
        config = super(RejectionAtEfficiency, self).get_config()
        config.update({'efficiency': self.efficiency,
                      'label_id': self.label_id})
        return config

    def result(self):
        return 1 / super(RejectionAtEfficiency, self).result()


class EffectiveTaggingEfficiency(tf.keras.metrics.Metric):

    def __init__(self, bins: List[float] = [0, 0.1, 0.25, 0.5, 0.625, 0.75, 0.875, 1], threshold: float = 0.5, name='eff_tag_efficiency', **kwargs):
        super(EffectiveTaggingEfficiency, self).__init__(name=name, **kwargs)
        for value in bins:
            if value < 0 or value > 1:
                raise ValueError('The bins must be between 0 and 1.')
        self.bins = bins
        self.threshold = threshold
        self.bin_sums = self.add_weight(
            name='bin_sums', initializer='zeros', shape=(len(bins) - 1,))
        self.bin_counts = self.add_weight(
            name='bin_counts', initializer='zeros', shape=(len(bins) - 1,))

    def update_state(self, y_true: Union[tf.Tensor, np.ndarray], score: Union[tf.Tensor, np.ndarray], sample_weight: Union[tf.Tensor, np.ndarray] = None):
        """Accumulates the metric.

        Args:
            y_true (Union[tf.Tensor, np.ndarray]): The true labels.
            score (Union[tf.Tensor, np.ndarray]): The output of the model, i.e. number **between 0 and 1**.
        """
        score = tf.squeeze(score)
        y_true = tf.squeeze(y_true)
        dilusion_factor = tf.abs(2 * score - 1)
        indicies = tf.searchsorted(self.bins, dilusion_factor)
        pred = tf.cast(score > 0.5, tf.bool)
        wrong_tag = tf.cast(tf.not_equal(
            pred, tf.cast(y_true, tf.bool)), tf.float32)

        bin_counts = tf.cast(tf.math.bincount(
            indicies, minlength=len(self.bins)), tf.float32)
        bin_counts = bin_counts[1:]
        bin_sums = tf.math.bincount(
            indicies, weights=wrong_tag, minlength=len(self.bins))
        bin_sums = bin_sums[1:]
        self.bin_counts.assign_add(bin_counts)
        self.bin_sums.assign_add(bin_sums)

    def get_config(self):
        config = super(EffectiveTaggingEfficiency, self).get_config()
        config.update({'bins': self.bins, 'threshold': self.threshold})
        return config

    def result(self):
        mask = tf.where(self.bin_counts > 0, True, False)
        bin_counts = tf.boolean_mask(self.bin_counts, mask)
        bin_sums = tf.boolean_mask(self.bin_sums, mask)
        binned_wrong_tag_fraction = bin_sums / bin_counts
        eff = bin_counts / tf.reduce_sum(bin_counts)
        eff_tag_eff = tf.reduce_sum(
            eff * (1 - 2 * binned_wrong_tag_fraction)**2)
        return eff_tag_eff

    def reset_state(self):
        self.bin_sums.assign(tf.zeros_like(self.bin_sums))
        self.bin_counts.assign(tf.zeros_like(self.bin_counts))


class FixedWorkingPointBase(tf.keras.metrics.Metric):
    r"""
    Base class for metrics that calculate the efficiencies and threshold at a fixed working point, 
    i.e. one fixed efficiency with variables threshold.

    Args:
        working_point (float): The working point, i.e. the efficiency at which the threshold is calculated.
        num_thresholds (int): The number of thresholds to calculate for finding the threshold at the working point.
        name (str): The name of the metric. 
        dtype (tf.dtypes.DType): The data type of the metric.       
    """

    def __init__(self, working_point: float = 0.5,
                 num_thresholds: int = 200, name: Optional[str] = None,
                 dtype: Optional[tf.dtypes.DType] = None):
        super().__init__(name=name, dtype=dtype)
        self.working_point = working_point
        self.num_thresholds = num_thresholds

        self.true_positives = self.add_weight(name='true_positives', initializer='zeros', shape=(num_thresholds,))
        self.total_positives = self.add_weight(name='total_positives', initializer='zeros', shape=(1,))
        self.false_negatives = self.add_weight(name='false_negatives', initializer='zeros', shape=(num_thresholds,))
        self.total_negatives = self.add_weight(name='total_negatives', initializer='zeros', shape=(1,))

        thresholds = [
            (i + 1) * 1.0 / (num_thresholds - 1)
            for i in range(num_thresholds - 2)
        ]
        self.thresholds = [0.0] + thresholds + [1.0]

    def update_state(self, y_true: Union[tf.Tensor, np.ndarray], y_pred: Union[tf.Tensor, np.ndarray], sample_weight: Optional[Union[tf.Tensor, np.ndarray]] = None):
        """Accumulates the confusion matrix statistics. The true positives and false negatives are calculated for each threshold.
        The total positives and total negatives are calculated once.

        Args:
            y_true (Union[tf.Tensor, np.ndarray]): True labels.
            y_pred (Union[tf.Tensor, np.ndarray]): Predicted scores, i.e. the output of the model.
            sample_weight (Optional[Union[tf.Tensor, np.ndarray]], optional): Sample weights. Defaults to None.
        """
        y_pred = tf.cast(tf.expand_dims(y_pred, axis=1) > self.thresholds, tf.bool)
        y_true = tf.expand_dims(y_true, axis=1)
        positive_equals = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        negative_equals = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, False))
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            sample_weight = tf.expand_dims(sample_weight, axis=1)
            positive_equals = tf.multiply(tf.cast(positive_equals, tf.float32), sample_weight)
            negative_equals = tf.multiply(tf.cast(negative_equals, tf.float32), sample_weight)

        self.true_positives.assign_add(tf.reduce_sum(tf.cast(positive_equals, tf.float32), axis=0))
        self.false_negatives.assign_add(tf.reduce_sum(tf.cast(negative_equals, tf.float32), axis=0))

        positives = tf.cast(tf.equal(y_true, True), tf.float32)
        positives = tf.multiply(positives, sample_weight) if sample_weight is not None else positives
        negatives = tf.cast(tf.equal(y_true, False), tf.float32)
        negatives = tf.multiply(negatives, sample_weight) if sample_weight is not None else negatives
        self.total_positives.assign_add(tf.reduce_sum(positives, axis=0))
        self.total_negatives.assign_add(tf.reduce_sum(negatives, axis=0))

    def result(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Calculates the efficiencies and threshold at the working point.
        Both the fixed and variable efficiencies are calculated and returned
        to allow a check of the working point.

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: The fixed efficiency, the variable efficiency and the threshold at the working point.

        """
        efficiency_positivess = self.true_positives / self.total_positives
        efficiency_negatives = self.false_negatives / self.total_negatives

        result_index = self._find_index_of_threshold(efficiency_positivess, self.working_point)
        positive_at_wp = tf.gather(efficiency_positivess, result_index)
        negative_at_wp = tf.gather(efficiency_negatives, result_index)
        threshold_at_wp = tf.gather(self.thresholds, result_index)

        return positive_at_wp, negative_at_wp, threshold_at_wp

    def _find_index_of_threshold(self, efficiencies: tf.Tensor, working_point: float) -> tf.Tensor:
        """Finds the index of the threshold at the working point. """
        result_index = tf.math.squared_difference(efficiencies, working_point)
        result_index = tf.argmin(result_index, axis=0)
        return result_index

    def reset_state(self):
        """Resets the confusion matrix statistics."""
        self.true_positives.assign(tf.zeros_like(self.true_positives))
        self.total_positives.assign(tf.zeros_like(self.total_positives))
        self.false_negatives.assign(tf.zeros_like(self.false_negatives))
        self.total_negatives.assign(tf.zeros_like(self.total_negatives))
    
class BkgRejVsSigEff(FixedWorkingPointBase):
    def __init__(self, thresholds: Union[int, List] = 200, name: Optional[str] = 'bkg_rej_vs_sig_eff', dtype: Optional[tf.dtypes.DType] = None, fixed_label_id: Literal[0, 1] = 1):
        if isinstance(thresholds, int):
            th = thresholds
        else:
            th = len(thresholds)
        super().__init__(working_point=0., num_thresholds=th, name=name, dtype=dtype)
        if isinstance(thresholds, list):
            self.thresholds = thresholds
        self.fixed_label_id = fixed_label_id
        
    def result(self) -> Tuple[tf.Tensor, tf.Tensor]:
        efficiency_positivess = self.true_positives / self.total_positives
        efficiency_negatives = self.false_negatives / self.total_negatives
        
        if self.fixed_label_id == 1:
            sig_efficiency = efficiency_positivess
            bkg_rejection = 1.0 / (1 - efficiency_negatives)
        elif self.fixed_label_id == 0:
            sig_efficiency = efficiency_negatives
            bkg_rejection = 1.0 / (1 - efficiency_positivess)
        else:
            raise ValueError(f'fixed_label_id must be 0 or 1, but is {self.fixed_label_id}')

        return sig_efficiency, bkg_rejection
        


class EfficiencyAtFixedWorkingPoint(FixedWorkingPointBase):
    """Calculates the efficiency at a fixed working point. The working point is defined by the user.

    Args:
        working_point (float): The working point. Defaults to 0.5.
        fixed_label_id (Literal[0, 1], optional): The label id whose efficiency is fixed. Defaults to 1.
        returned_label_id (Literal[0, 1], optional): The label id whose efficiency is returned. Defaults to 0.
        num_thresholds (int, optional): The number of thresholds to use for the estimation. Defaults to 200.
        name (Optional[str], optional): The name of the metric. Defaults to 'efficiency_at_fixed_wp'.
        dtype (Optional[tf.dtypes.DType], optional): The data type of the metric. Defaults to None.
    """

    def __init__(self, working_point: float = 0.5, fixed_label_id: Literal[0, 1] = 1, returned_label_id: Literal[0, 1] = 0,
                 num_thresholds: int = 200, name: Optional[str] = 'efficiency_at_fixed_wp', dtype: Optional[tf.dtypes.DType] = None):

        super().__init__(working_point=working_point, num_thresholds=num_thresholds, name=name, dtype=dtype)
        self.fixed_label_id = fixed_label_id
        self.returned_label_id = returned_label_id

    def result(self) -> tf.Tensor:
        """Calculates the efficiency at the working point.

        Raises:
            ValueError: If the fixed_label_id is not 0 or 1.

        Returns:
            tf.Tensor: The efficiency at the working point.
        """
        efficiency_positivess = self.true_positives / self.total_positives
        efficiency_negatives = self.false_negatives / self.total_negatives

        if self.fixed_label_id == 1:
            fixed_efficiency = efficiency_positivess
        elif self.fixed_label_id == 0:
            fixed_efficiency = efficiency_negatives
        else:
            raise ValueError(f'fixed_label_id must be 0 or 1, but is {self.fixed_label_id}')

        closest_index = self._find_index_of_threshold(fixed_efficiency, self.working_point)

        if self.returned_label_id == 1:
            efficiency_at_wp = tf.gather(efficiency_positivess, closest_index)
        elif self.returned_label_id == 0:
            efficiency_at_wp = tf.gather(efficiency_negatives, closest_index)

        return efficiency_at_wp


class RejectionAtFixedWorkingPoint(EfficiencyAtFixedWorkingPoint):
    """Calculates the rejection at a fixed working point. The working point is defined by the user.

    Args:
        working_point (float): The working point. Defaults to 0.5.
        fixed_label_id (Literal[0, 1], optional): The label id whose efficiency is fixed. Defaults to 1.
        returned_label_id (Literal[0, 1], optional): The label id whose efficiency is returned. Defaults to 0.
        num_thresholds (int, optional): The number of thresholds to use for the estimation. Defaults to 200.
        name (Optional[str], optional): The name of the metric. Defaults to 'efficiency_at_fixed_wp'.
        dtype (Optional[tf.dtypes.DType], optional): The data type of the metric. Defaults to None.
    """

    def __init__(self, working_point: float = 0.5, fixed_label_id: Literal[0, 1] = 1, returned_label_id: Literal[0, 1] = 0,
                 num_thresholds: int = 200, name: Optional[str] = 'rejection_at_fixed_wp', dtype: Optional[tf.dtypes.DType] = None):
        super().__init__(working_point=working_point, fixed_label_id=fixed_label_id, returned_label_id=returned_label_id,
                         num_thresholds=num_thresholds, name=name, dtype=dtype)

    def result(self) -> tf.Tensor:
        """Calculates the rejection at the working point.

        Returns:
            tf.Tensor: The rejection at the working point.
        """

        return 1.0 / (1 - super().result())


class ThresholdAtFixedWorkingPoint(FixedWorkingPointBase):
    """Calculates the threshold at a fixed working point. The working point is defined by the user.

    Args:
        working_point (float): The working point. Defaults to 0.5.
        fixed_label_id (Literal[0, 1], optional): The label id whose efficiency is fixed. Defaults to 1.
        num_thresholds (int, optional): The number of thresholds to use for the estimation. Defaults to 200.
        name (Optional[str], optional): The name of the metric. Defaults to 'threshold_at_fixed_wp'.
        dtype (Optional[tf.dtypes.DType], optional): The data type of the metric. Defaults to None.
    """

    def __init__(self, working_point: float = 0.5, num_thresholds: int = 200, fixed_label_id: Literal[0, 1] = 1, name: Optional[str] = 'threshold_at_fixed_efficiency', dtype: Optional[tf.dtypes.DType] = None):
        super().__init__(working_point=working_point, num_thresholds=num_thresholds, name=name, dtype=dtype)
        self.fixed_label_id = fixed_label_id

    def result(self) -> tf.Tensor:
        """Calculates the threshold at the working point.

        Raises:
            ValueError: If the fixed_label_id is not 0 or 1.

        Returns:
            tf.Tensor: The threshold at the working point.
        """

        if self.fixed_label_id == 1:
            efficiencies = self.true_positives / self.total_positives
        elif self.fixed_label_id == 0:
            efficiencies = self.false_negatives / self.total_negatives
        else:
            raise ValueError(f'fixed_label_id must be 0 or 1, but is {self.fixed_label_id}')

        closest_index = self._find_index_of_threshold(efficiencies, self.working_point)

        return tf.gather(self.thresholds, closest_index)


def get_metrics(threshold: float = 0.5) -> List[tf.keras.metrics.Metric]:
    """Returns a list of metrics.

    Args:
        threshold (float, optional): The threshold for the prediction. Defaults to 0.5.

    Returns:
        List[tf.keras.metrics.Metric]: The list of selected metrics.
    """
    metrics = [
        tf.keras.metrics.BinaryAccuracy(
            name='binary_accuracy', threshold=0.5), # fix the threshold to 0.5, other thresholds are useless for this metric
        BinaryEfficiency(name='gluon_efficiency',
                         label_id=0, threshold=threshold),
        BinaryEfficiency(name='quark_efficiency',
                         label_id=1, threshold=threshold),
        BinaryRejection(name='gluon_rejection',
                        label_id=0, threshold=threshold),
        BinaryRejection(name='quark_rejection',
                        label_id=1, threshold=threshold),
        EfficiencyAtFixedWorkingPoint(name='gluon_efficiency_at_quark_50wp',
                                      fixed_label_id=1, working_point=0.5, returned_label_id=0),
        EfficiencyAtFixedWorkingPoint(name='quark_efficiency_at_quark_50wp',
                                      fixed_label_id=1, working_point=0.5, returned_label_id=1),
        RejectionAtFixedWorkingPoint(name='gluon_rejection_at_quark_50wp',
                                     fixed_label_id=1, working_point=0.5, returned_label_id=0),
        ThresholdAtFixedWorkingPoint(name='threshold_at_fixed_quark_50wp',
                                     fixed_label_id=1, working_point=0.5),
        EfficiencyAtFixedWorkingPoint(name='gluon_efficiency_at_quark_80wp',
                                      fixed_label_id=1, working_point=0.8, returned_label_id=0),
        EfficiencyAtFixedWorkingPoint(name='quark_efficiency_at_quark_80wp',
                                      fixed_label_id=1, working_point=0.8, returned_label_id=1),
        RejectionAtFixedWorkingPoint(name='gluon_rejection_at_quark_80wp',
                                     fixed_label_id=1, working_point=0.8, returned_label_id=0),
        ThresholdAtFixedWorkingPoint(name='threshold_at_fixed_quark_80wp',
                                     fixed_label_id=1, working_point=0.8),
        # EffectiveTaggingEfficiency(
        #     name='effective_tagging_efficiency', threshold=threshold),
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
        metric.reset_state()
        metric.update_state(y_true, score, sample_weight=weights)
        result = metric.result().numpy()
        results[metric.name] = result
    return results
