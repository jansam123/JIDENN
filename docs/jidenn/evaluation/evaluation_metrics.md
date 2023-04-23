Module jidenn.evaluation.evaluation_metrics
===========================================
Module containing custom metrics for evaluating models mainly used in HEP applications.

Functions
---------

    
`calculate_metrics(y_true: numpy.ndarray, score: numpy.ndarray, threshold: float = 0.5, weights: Optional[numpy.ndarray] = None) ‑> Dict[str, float]`
:   Calculates the metrics.
    Args:
        y_true (np.ndarray): The true labels.
        score (np.ndarray): The output scores of the model.
        threshold (float, optional): The threshold for the prediction. Defaults to 0.5.
        weights (np.ndarray, optional): The sample weights. Defaults to None.
    Returns:
        Dict[str, float]: The dictionary of metric names and values.

    
`get_metrics(threshold: float = 0.5) ‑> List[keras.metrics.base_metric.Metric]`
:   Returns a list of metrics.
    
    Args:
        threshold (float, optional): The threshold for the prediction. Defaults to 0.5.
    
    Returns:
        List[tf.keras.metrics.Metric]: The list of selected metrics.

Classes
-------

`BinaryEfficiency(label_id: Literal[0, 1] = 1, threshold=0.5, name='efficiency')`
:   Binary Efficiency metric.
    It is defined as
    $$\varepsilon_i=\frac{T_i}{T_i+F_i}$$
    where $T_i$ is the number of correctly classified data of class $i$ 
    and $F_i$ is the number of incorrectly classified data of class $i$.
    
    If $i = 1$, then it is the efficiency is called true positive rate (TPR).
    
    Args:
        label_id (int, optional): The label id for which the efficiency is calculated. Defaults to 1.
        threshold (float, optional): The threshold for the prediction. Defaults to 0.5.
        name (str, optional): The name of the metric. Defaults to 'efficiency'.

    ### Ancestors (in MRO)

    * keras.metrics.base_metric.Metric
    * keras.engine.base_layer.Layer
    * tensorflow.python.module.module.Module
    * tensorflow.python.trackable.autotrackable.AutoTrackable
    * tensorflow.python.trackable.base.Trackable
    * keras.utils.version_utils.LayerVersionSelector

    ### Methods

    `get_config(self)`
    :   Returns the serializable config of the metric.

    `reset_states(self)`
    :

    `result(self)`
    :   Computes and returns the scalar metric value tensor or a dict of
        scalars.
        
        Result computation is an idempotent operation that simply calculates the
        metric value using the state variables.
        
        Returns:
          A scalar tensor, or a dictionary of scalar tensors.

    `update_state(self, y_true: Union[tensorflow.python.framework.ops.Tensor, numpy.ndarray], y_pred: Union[tensorflow.python.framework.ops.Tensor, numpy.ndarray], sample_weight: Union[tensorflow.python.framework.ops.Tensor, numpy.ndarray] = None)`
    :   Accumulates the efficiency.
        
        Args:
            y_true (tf.Tensor): The true labels.
            y_pred (tf.Tensor): The predicted labels.
            sample_weight (tf.Tensor, optional): The sample weights. Defaults to None.

`BinaryRejection(label_id: Literal[0, 1] = 1, threshold=0.5, name='rejection')`
:   Binary Rejection metric.
    It is defined as
    $$\varepsilon_i^{-1}=\frac{T_i+F_i}{T_i}$$
    where $T_i$ is the number of correctly classified data of class $i$ 
    and $F_i$ is the number of incorrectly classified data of class $i$.
    
    Args:
        label_id (int, optional): The label id for which the efficiency is calculated. Defaults to 1.
        threshold (float, optional): The threshold for the prediction. Defaults to 0.5.
        name (str, optional): The name of the metric. Defaults to 'rejection'.

    ### Ancestors (in MRO)

    * keras.metrics.base_metric.Metric
    * keras.engine.base_layer.Layer
    * tensorflow.python.module.module.Module
    * tensorflow.python.trackable.autotrackable.AutoTrackable
    * tensorflow.python.trackable.base.Trackable
    * keras.utils.version_utils.LayerVersionSelector

    ### Methods

    `get_config(self)`
    :   Returns the serializable config of the metric.

    `reset_states(self)`
    :

    `result(self)`
    :   Computes and returns the scalar metric value tensor or a dict of
        scalars.
        
        Result computation is an idempotent operation that simply calculates the
        metric value using the state variables.
        
        Returns:
          A scalar tensor, or a dictionary of scalar tensors.

    `update_state(self, y_true: Union[tensorflow.python.framework.ops.Tensor, numpy.ndarray], y_pred: Union[tensorflow.python.framework.ops.Tensor, numpy.ndarray], sample_weight: Union[tensorflow.python.framework.ops.Tensor, numpy.ndarray] = None)`
    :   Accumulates the efficiency.
        
        Args:
            y_true (tf.Tensor): The true labels.
            y_pred (tf.Tensor): The predicted labels.
            sample_weight (tf.Tensor, optional): The sample weights. Defaults to None.

`RejectionAtEfficiency(efficiency: float = 0.5, label_id: Literal[0, 1] = 1, name='rejection_at_efficiency')`
:   Rejection at efficiency metric.
    in this case the threshold is chosen such that the efficiency of one class is equal to the given `efficiency`.
    
    Args:
        efficiency (float, optional): The efficiency. Defaults to 0.5.
        label_id (int, optional): The label id for which the efficiency is calculated. Defaults to 1.
        name (str, optional): The name of the metric. Defaults to 'rejection_at_efficiency'.

    ### Ancestors (in MRO)

    * keras.metrics.metrics.SpecificityAtSensitivity
    * keras.metrics.metrics.SensitivitySpecificityBase
    * keras.metrics.base_metric.Metric
    * keras.engine.base_layer.Layer
    * tensorflow.python.module.module.Module
    * tensorflow.python.trackable.autotrackable.AutoTrackable
    * tensorflow.python.trackable.base.Trackable
    * keras.utils.version_utils.LayerVersionSelector

    ### Methods

    `get_config(self)`
    :   Returns the serializable config of the metric.

    `result(self)`
    :   Computes and returns the scalar metric value tensor or a dict of
        scalars.
        
        Result computation is an idempotent operation that simply calculates the
        metric value using the state variables.
        
        Returns:
          A scalar tensor, or a dictionary of scalar tensors.

    `update_state(self, y_true: Union[tensorflow.python.framework.ops.Tensor, numpy.ndarray], y_pred: Union[tensorflow.python.framework.ops.Tensor, numpy.ndarray], sample_weight: Union[tensorflow.python.framework.ops.Tensor, numpy.ndarray] = None)`
    :   Accumulates the efficiency.
        
        Args:
            y_true (tf.Tensor): The true labels.
            y_pred (tf.Tensor): The predicted labels.
            sample_weight (tf.Tensor, optional): The sample weights. Defaults to None.