"""
Module for initializing models using the configuration file.
The config options are explicitly passed to the model classes `__init__` method.
For each model, a function is defined that returns an instance of the model class.
The mapping from string names of the models in the config file is then mapped to the corresponding function.
"""

import tensorflow as tf
import keras
from typing import Tuple, Optional, Callable, List, Literal, Union, Any

from jidenn.config import model_config

from ..models.FC import FCModel
from ..models.Highway import HighwayModel
from ..models.Transformer import TransformerModel
from ..models.ParT import ParTModel
from ..models.DeParT import DeParTModel
from ..models.PFN import PFNModel
from ..models.EFN import EFNModel
# from ..models.BDT import bdt_model
from ..models.ParticleNet import ParticleNetModel
from ..models.DeParT2 import DeParT2Model


def get_activation(activation: Literal['relu', 'gelu', 'tanh', 'swish']) -> Callable[[tf.Tensor], tf.Tensor]:
    """Map string names of activation functions to the corresponding function.

    Args:
        activation (str): Name of the activation function. One of ['relu', 'gelu', 'tanh', 'swish'].
    Returns:
        Callable[[tf.Tensor], tf.Tensor]: Activation function.
    """
    if activation == 'relu':
        return keras.activations.relu
    elif activation == 'gelu':
        return keras.activations.gelu
    elif activation == 'tanh':
        return keras.activations.tanh
    elif activation == 'swish':
        return keras.activations.swish
    else:
        raise NotImplementedError(f'Activation {activation} not supported.')


def get_FC_model(input_size: int,
                 output_layer: keras.layers.Layer,
                 args_model: model_config.FC,
                 preprocess: Optional[keras.layers.Layer] = None) -> FCModel:
    """Get an instance of the fully-connected model.

    Args:
        input_size (int): Input size of the model.
        output_layer (keras.layers.Layer): Output layer of the model.
        args_model (jidenn.config.model_config.FC): Model configuration.
        preprocess (Optional[keras.layers.Layer], optional): Preprocessing layer. Defaults to None.

    Returns:
        FCModel: Fully-connected model.
    """

    return FCModel(
        layer_size=args_model.layer_size,
        num_layers=args_model.num_layers,
        dropout=args_model.dropout,
        input_size=input_size,
        output_layer=output_layer,
        activation=get_activation(args_model.activation),
        preprocess=preprocess)


def get_highway_model(input_size: int,
                      output_layer: keras.layers.Layer,
                      args_model: model_config.Highway,
                      preprocess: Optional[keras.layers.Layer] = None) -> HighwayModel:
    """Get an instance of the highway model.

    Args:
        input_size (int): Input size of the model.
        output_layer (keras.layers.Layer): Output layer of the model.
        args_model (jidenn.config.model_config.Highway): Model configuration.
        preprocess (Optional[keras.layers.Layer], optional): Preprocessing layer. Defaults to None.

    Returns:
        HighwayModel: Highway model.
    """

    return HighwayModel(
        layer_size=args_model.layer_size,
        num_layers=args_model.num_layers,
        dropout=args_model.dropout,
        input_size=input_size,
        output_layer=output_layer,
        activation=get_activation(args_model.activation),
        preprocess=preprocess)


def get_pfn_model(input_size: Tuple[None, int],
                  output_layer: keras.layers.Layer,
                  args_model: model_config.PFN,
                  preprocess: Optional[keras.layers.Layer] = None) -> PFNModel:
    """Get an instance of the PFN model.

    Args:
        input_size (Tuple[None, int]): Input size of the model.
        output_layer (keras.layers.Layer): Output layer of the model.
        args_model (model_config.PFN): Model configuration.
        preprocess (Optional[keras.layers.Layer], optional): Preprocessing layer. Defaults to None.

    Returns:
        PFNModel: PFN model.
    """

    return PFNModel(
        input_shape=input_size,
        Phi_sizes=args_model.Phi_sizes,
        Phi_dropout=args_model.Phi_dropout,
        F_sizes=args_model.F_sizes,
        F_dropout=args_model.F_dropout,
        Phi_backbone=args_model.Phi_backbone,
        activation=get_activation(args_model.activation),
        batch_norm=args_model.batch_norm,
        preprocess=preprocess,
        output_layer=output_layer)


def get_efn_model(input_size: Tuple[None, int],
                  output_layer: keras.layers.Layer,
                  args_model: model_config.EFN,
                  preprocess: Optional[keras.layers.Layer] = None) -> EFNModel:
    """Get an instance of the EFN model.

    Args:
        input_size (Tuple[None, int]): Input size of the model.
        output_layer (keras.layers.Layer): Output layer of the model.
        args_model (model_config.EFN): Model configuration.
        preprocess (Optional[keras.layers.Layer], optional): Preprocessing layer. Defaults to None.  

    Returns:
        EFNModel: EFN model.
    """

    return EFNModel(
        input_shape=input_size,
        Phi_sizes=args_model.Phi_sizes,
        F_sizes=args_model.F_sizes,
        Phi_backbone=args_model.Phi_backbone,
        activation=get_activation(args_model.activation),
        batch_norm=args_model.batch_norm,
        preprocess=preprocess,
        output_layer=output_layer)


def get_transformer_model(input_size: Tuple[None, int],
                          output_layer: keras.layers.Layer,
                          args_model: model_config.Transformer,
                          preprocess: Optional[keras.layers.Layer] = None) -> TransformerModel:
    """Get an instance of the transformer model.

    Args:
        input_size (Tuple[None, int]): Input size of the model.
        output_layer (keras.layers.Layer): Output layer of the model.
        args_model (model_config.Transformer): Model configuration.
        preprocess (Optional[keras.layers.Layer], optional): Preprocessing layer. Defaults to None.  

    Returns:
        TransformerModel: Transformer model.
    """

    return TransformerModel(
        embed_dim=args_model.embed_dim,
        embed_layers=args_model.embed_layers,
        dropout=args_model.dropout,
        expansion=args_model.expansion,
        heads=args_model.heads,
        self_attn_layers=args_model.self_attn_layers,
        activation=get_activation(args_model.activation),
        input_shape=input_size,
        output_layer=output_layer,
        preprocess=preprocess)


def get_part_model(input_size: Union[Tuple[None, int], Tuple[Tuple[None, int], Tuple[None, None, int]]],
                   output_layer: keras.layers.Layer,
                   args_model: model_config.ParT,
                   preprocess: Optional[keras.layers.Layer] = None) -> ParTModel:
    """Get an instance of the ParT model.

    Args:
        input_size (Union[Tuple[None, int], Tuple[Tuple[None, int], Tuple[None, None, int]]]): Input size of the model.
        output_layer (keras.layers.Layer): Output layer of the model.
        args_model (model_config.ParT): Model configuration.
        preprocess (Optional[keras.layers.Layer], optional): Preprocessing layer. Defaults to None.

    Returns:
        ParTModel: ParT model.
    """

    return ParTModel(
        input_shape=input_size,
        output_layer=output_layer,
        #
        embed_dim=args_model.embed_dim,
        embed_layers=args_model.embed_layers,
        self_attn_layers=args_model.self_attn_layers,
        class_attn_layers=args_model.class_attn_layers,
        expansion=args_model.expansion,
        heads=args_model.heads,
        dropout=args_model.dropout,
        interaction_embed_layers=args_model.interaction_embedding_layers,
        interaction_embed_layer_size=args_model.interaction_embedding_layer_size,
        #
        preprocess=preprocess,
        activation=get_activation(args_model.activation))


def get_depart_model(input_size: Union[Tuple[None, int], Tuple[Tuple[None, int], Tuple[None, None, int]]],
                     output_layer: keras.layers.Layer,
                     args_model: model_config.DeParT,
                     preprocess: Optional[keras.layers.Layer] = None) -> DeParTModel:
    """Get an instance of the DeParT model.

    Args:
        input_size (Union[Tuple[None, int], Tuple[Tuple[None, int], Tuple[None, None, int]]]): Input size of the model.
        output_layer (keras.layers.Layer): Output layer of the model.
        args_model (model_config.DeParT): Model configuration.
        preprocess (Optional[keras.layers.Layer], optional): Preprocessing layer. Defaults to None.

    Returns:
        DeParTModel: DeParT model.
    """

    return DeParTModel(
        input_shape=input_size,
        output_layer=output_layer,
        #
        embed_dim=args_model.embed_dim,
        embed_layers=args_model.embed_layers,
        self_attn_layers=args_model.self_attn_layers,
        class_attn_layers=args_model.class_attn_layers,
        expansion=args_model.expansion,
        heads=args_model.heads,
        dropout=args_model.dropout,
        layer_scale_init_value=args_model.layer_scale_init_value,
        stochastic_depth_drop_rate=args_model.stochastic_depth_drop_rate,
        class_dropout=args_model.class_dropout,
        class_stochastic_depth_drop_rate=args_model.class_stochastic_depth_drop_rate,
        interaction_embed_layers=args_model.interaction_embedding_layers,
        interaction_embed_layer_size=args_model.interaction_embedding_layer_size,
        #
        preprocess=preprocess,
        activation=get_activation(args_model.activation))

def get_depart2_model(input_size: Union[Tuple[None, int], Tuple[Tuple[None, int], Tuple[None, None, int]]],
                     output_layer: keras.layers.Layer,
                     args_model: model_config.DeParT,
                     preprocess: Optional[keras.layers.Layer] = None) -> DeParTModel:
    """Get an instance of the DeParT model.

    Args:
        input_size (Union[Tuple[None, int], Tuple[Tuple[None, int], Tuple[None, None, int]]]): Input size of the model.
        output_layer (keras.layers.Layer): Output layer of the model.
        args_model (model_config.DeParT): Model configuration.
        preprocess (Optional[keras.layers.Layer], optional): Preprocessing layer. Defaults to None.

    Returns:
        DeParTModel: DeParT model.
    """

    return DeParT2Model(
        input_shape=input_size,
        output_layer=output_layer,
        #
        embed_dim=args_model.embed_dim,
        embed_layers=args_model.embed_layers,
        self_attn_layers=args_model.self_attn_layers,
        class_attn_layers=args_model.class_attn_layers,
        expansion=args_model.expansion,
        heads=args_model.heads,
        dropout=args_model.dropout,
        layer_scale_init_value=args_model.layer_scale_init_value,
        stochastic_depth_drop_rate=args_model.stochastic_depth_drop_rate,
        class_dropout=args_model.class_dropout,
        class_stochastic_depth_drop_rate=args_model.class_stochastic_depth_drop_rate,
        interaction_embed_layers=args_model.interaction_embedding_layers,
        interaction_embed_layer_size=args_model.interaction_embedding_layer_size,
        #
        preprocess=preprocess,
        activation=get_activation(args_model.activation))


def get_particlenet_model(input_size: Tuple[Tuple[int, int], Tuple[int, int]],
                          output_layer: keras.layers.Layer,
                          args_model: model_config.ParticleNet,
                          preprocess: Optional[keras.layers.Layer] = None) -> ParticleNetModel:

    return ParticleNetModel(
        input_shape=input_size,
        output_layer=output_layer,
        activation=get_activation(args_model.activation),
        preprocess=preprocess,
        pooling=args_model.pooling,
        fc_layers=list(args_model.fc_layers),
        fc_dropout=list(args_model.fc_dropout),
        edge_knn=list(args_model.edge_knn),
        edge_layers=list([list(layer) for layer in args_model.edge_layers]),
    )


def get_bdt_model(input_size: int,
                  output_layer: keras.layers.Layer,
                  args_model: model_config.BDT,
                  preprocess: Optional[keras.layers.Layer] = None):
    """Get an instance of the BDT model.

    Args:
        args_model (model_config.BDT): Model configuration.
        input_size (int): Input size of the model. UNUSED
        output_layer (keras.layers.Layer): Output layer of the model. UNUSED
        preprocess (Optional[keras.layers.Layer], optional): Preprocessing layer. Defaults to None. UNUSED

    Returns:
        tfdf.keras.GradientBoostedTreesModel: BDT model.
    """
    raise NotImplementedError('BDT model not supported.')
    # return bdt_model(args_model)


def model_getter_lookup(model_name: Literal['fc', 'highway', 'pfn', 'efn', 'transformer', 'part', 'depart', 'bdt']
                        ) -> Callable[[Any, keras.layers.Layer, model_config.Model, Optional[keras.layers.Layer]], keras.Model]:
    """Get a model getter function.

    Args:
        model_name (str): Name of the model. Options are 'fc', 'highway', 'pfn', 'efn', 'transformer', 'part', 'depart', 'bdt'.

    Returns:
        Callable[[Any, keras.layers.Layer, model_config.Model, Optional[keras.layers.Layer]], keras.Model]: Model getter function.
    """

    lookup_model = {'fc': get_FC_model,
                    'highway': get_highway_model,
                    'pfn': get_pfn_model,
                    'efn': get_efn_model,
                    'transformer': get_transformer_model,
                    'particlenet': get_particlenet_model,
                    'part': get_part_model,
                    'depart': get_depart_model,
                    'depart2': get_depart2_model,
                    'bdt': get_bdt_model, }

    if model_name not in lookup_model:
        raise ValueError(f'Unknown model {model_name}. Possible options are {list(lookup_model.keys())}')

    return lookup_model[model_name]
