Module jidenn.models.PFN
========================
Implementation of the Particle Flow Network (PFN) model.
See https://energyflow.network for original implementation or
the paper https://arxiv.org/abs/1810.05165 for more details.

The input are the jet constituents, i.e. the particles in the jet foorming an input shape of `(batch_size, num_particles, num_features)`.
The model is then executed as follows:
$$ \mathrm{output} = F\left(\sum_i  \Phi{(\eta_i,\phi_i, \pmb{E_i})}\right) $$

The F function is any function, which is applied to the summed features. The default is a fully-connected network.
The mapping Phi can be a fully-connected network or a convolutional network. The default is a fully-connected network.

The PFN is a modification of the EFN, where the energy part is also mapped with the Phi function.
However the model then **violates Infrared and Collinear safety**.

![EFN_PFN](../../../diagrams/pfn_efn.png)
On the left is the PFN model, on the right the EFN model.

Classes
-------

`PFNModel(input_shape: Tuple[None, int], Phi_sizes: List[int], F_sizes: List[int], output_layer: keras.engine.base_layer.Layer, activation: Callable[[tensorflow.python.framework.ops.Tensor], tensorflow.python.framework.ops.Tensor], Phi_backbone: Literal['cnn', 'fc'] = 'fc', batch_norm: bool = False, Phi_dropout: Optional[float] = None, F_dropout: Optional[float] = None, preprocess: Optional[keras.engine.base_layer.Layer] = None)`
:   Implements the Particle Flow Network (PFN) model.
    
    The expected input shape is `(batch_size, num_particles, num_features)`, where the second dimension is ragged.
    
    Args:
        input_shape (Tuple[None, int]): The shape of the input.
        Phi_sizes (List[int]): The sizes of the hidden layers of the Phi function.
        F_sizes (List[int]): The sizes of the hidden layers of the F function.
        output_layer (tf.keras.layers.Layer): The output layer of the model.
        activation (Callable[[tf.Tensor], tf.Tensor]) The activation function. 
        Phi_backbone (str, optional): The backbone of the Phi function. Options are "fc" for a fully-connected network 
            and "cnn" for a convolutional network. Defaults to "fc".
            This option is omitted from the `jidenn.config.model_config.PFN` as it is not used in the paper.
        batch_norm (bool, optional): Whether to use batch normalization. Defaults to False.
            This option is omitted from the `jidenn.config.model_config.PFN` as it is not used in the paper.
        Phi_dropout (float, optional): The dropout rate of the Phi function. Defaults to None.
        F_dropout (float, optional): The dropout rate of the F function. Defaults to None.
        preprocess (tf.keras.layers.Layer, optional): The preprocessing layer. Defaults to None.

    ### Ancestors (in MRO)

    * keras.engine.training.Model
    * keras.engine.base_layer.Layer
    * tensorflow.python.module.module.Module
    * tensorflow.python.trackable.autotrackable.AutoTrackable
    * tensorflow.python.trackable.base.Trackable
    * keras.utils.version_utils.LayerVersionSelector
    * keras.utils.version_utils.ModelVersionSelector

    ### Methods

    `cnn_Phi(self, inputs: tensorflow.python.framework.ops.Tensor) ‑> tensorflow.python.framework.ops.Tensor`
    :   Convolutional Phi mapping.
        
        Args:
            inputs (tf.Tensor): The input tensor of shape `(batch_size, num_particles, num_features)`.
        
        Returns:
            tf.Tensor: The output tensor of shape `(batch_size, num_particles, Phi_sizes[-1])`.

    `fc_F(self, inputs: tensorflow.python.framework.ops.Tensor) ‑> tensorflow.python.framework.ops.Tensor`
    :   Fully connected F mapping.
        
        Args:
            inputs (tf.Tensor): The input tensor of shape `(batch_size, num_features)`.
        
        Returns:
            tf.Tensor: The output tensor of shape `(batch_size, F_sizes[-1])`.

    `fc_Phi(self, inputs: tensorflow.python.framework.ops.Tensor) ‑> tensorflow.python.framework.ops.Tensor`
    :   Fully connected Phi mapping.
        
        Args:
            inputs (tf.Tensor): The input tensor of shape `(batch_size, num_particles, num_features)`.
        
        Returns:
            tf.Tensor: The output tensor of shape `(batch_size, num_particles, Phi_sizes[-1])`.