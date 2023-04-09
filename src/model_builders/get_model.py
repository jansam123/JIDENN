import tensorflow as tf
import tensorflow_addons as tfa
from typing import Union, Tuple, Optional, Callable, List

from src.config import model_config as model_cfg

from ..models.FC import FCModel
from ..models.Highway import HighwayModel
from ..models.Transformer import TransformerModel
from ..models.ParT import ParTModel
from ..models.DeParT import DeParTModel
from ..models.PFN import PFNModel
from ..models.EFN import EFNModel
from ..models.BDT import bdt_model


def get_metrics(num_labels: int) -> List[tf.keras.metrics.Metric]:
    metrics = [tf.keras.metrics.CategoricalAccuracy() if num_labels > 2 else tf.keras.metrics.BinaryAccuracy(),
               tf.keras.metrics.AUC()]
    return metrics


def get_loss(num_labels: int, label_smoothing: float) -> tf.keras.losses.Loss:
    if num_labels > 2:
        return tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    else:
        return tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)


def get_output_layer(num_labels: int) -> tf.keras.layers.Layer:
    if num_labels > 2:
        return tf.keras.layers.Dense(num_labels, activation=tf.nn.softmax)
    else:
        return tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)


def get_activation(activation: str) -> Callable:
    if activation == 'relu':
        return tf.nn.relu
    elif activation == 'gelu':
        return tfa.activations.gelu
    elif activation == 'tanh':
        return tf.nn.tanh
    elif activation == 'swish':
        return tf.keras.activations.swish
    else:
        raise NotImplementedError(f'Activation {activation} not supported.')


def get_FC_model(input_size,
                 output_layer,
                 args_model: model_cfg.BasicFC,
                 preprocess: Optional[tf.keras.layers.Layer] = None):

    return FCModel(
        layer_size=args_model.layer_size,
        num_layers=args_model.num_layers,
        dropout=args_model.dropout,
        input_size=input_size,
        output_layer=output_layer,
        activation=get_activation(args_model.activation),
        preprocess=preprocess)


def get_highway_model(input_size,
                      output_layer: tf.keras.layers.Layer,
                      args_model: model_cfg.Highway,
                      preprocess: Optional[tf.keras.layers.Layer] = None):

    return HighwayModel(
        layer_size=args_model.layer_size,
        num_layers=args_model.num_layers,
        dropout=args_model.dropout,
        input_size=input_size,
        output_layer=output_layer,
        activation=get_activation(args_model.activation),
        preprocess=preprocess)


def get_pfn_model(input_size,
                  output_layer: tf.keras.layers.Layer,
                  args_model: model_cfg.PFN,
                  preprocess: Optional[tf.keras.layers.Layer] = None):

    return PFNModel(
        input_shape=input_size,
        Phi_sizes=args_model.Phi_sizes,
        F_sizes=args_model.F_sizes,
        Phi_backbone=args_model.Phi_backbone,
        batch_norm=args_model.batch_norm,
        preprocess=preprocess,
        output_layer=output_layer)


def get_efn_model(input_size,
                  output_layer: tf.keras.layers.Layer,
                  args_model: model_cfg.EFN,
                  preprocess: Optional[tf.keras.layers.Layer] = None):

    return EFNModel(
        input_shape=input_size,
        Phi_sizes=args_model.Phi_sizes,
        F_sizes=args_model.F_sizes,
        Phi_backbone=args_model.Phi_backbone,
        batch_norm=args_model.batch_norm,
        preprocess=preprocess,
        output_layer=output_layer)


def get_transformer_model(input_size: Tuple[int],
                          output_layer: tf.keras.layers.Layer,
                          args_model: model_cfg.Transformer,
                          preprocess: Optional[tf.keras.layers.Layer] = None):

    return TransformerModel(
        embedding_dim=args_model.embed_dim,
        num_embeding_layers=args_model.num_embed_layers,
        dropout=args_model.dropout,
        expansion=args_model.expansion,
        heads=args_model.heads,
        layers=args_model.layers,
        activation=get_activation(args_model.activation),
        input_shape=input_size,
        output_layer=output_layer,
        preprocess=preprocess)


def get_part_model(input_size,
                   output_layer: tf.keras.layers.Layer,
                   args_model: model_cfg.ParT,
                   preprocess: Optional[tf.keras.layers.Layer] = None):

    return ParTModel(
        input_shape=input_size,
        output_layer=output_layer,
        #
        embedding_dim=args_model.embed_dim,
        num_embeding_layers=args_model.num_embed_layers,
        selfattn_block_layers=args_model.particle_block_layers,
        class_block_layers=args_model.class_block_layers,
        expansion=args_model.expansion,
        heads=args_model.heads,
        dropout=args_model.particle_block_dropout,
        interaction_embedding_num_layers=args_model.interaction_embedding_num_layers,
        interaction_embedding_layer_size=args_model.interaction_embedding_layer_size,
        #
        preprocess=preprocess,
        activation=get_activation(args_model.activation))


def get_depart_model(input_size,
                     output_layer: tf.keras.layers.Layer,
                     args_model: model_cfg.DeParT,
                     preprocess: Optional[tf.keras.layers.Layer] = None):

    return DeParTModel(
        input_shape=input_size,
        output_layer=output_layer,
        #
        embedding_dim=args_model.embed_dim,
        num_embeding_layers=args_model.num_embed_layers,
        layers=args_model.layers,
        class_layers=args_model.class_layers,
        expansion=args_model.expansion,
        heads=args_model.heads,
        dropout=args_model.dropout,
        layer_scale_init_value=args_model.layer_scale_init_value,
        stochastic_depth_drop_rate=args_model.stochastic_depth_drop_rate,
        class_dropout=args_model.class_dropout,
        class_stochastic_depth_drop_rate=args_model.class_stochastic_depth_drop_rate,
        interaction_embedding_num_layers=args_model.interaction_embedding_num_layers,
        interaction_embedding_layer_size=args_model.interaction_embedding_layer_size,
        #
        preprocess=preprocess,
        activation=get_activation(args_model.activation))


def get_bdt_model(input_size,
                  output_layer: tf.keras.layers.Layer,
                  args_model: model_cfg.BDT,
                  preprocess: Optional[tf.keras.layers.Layer] = None):

    return bdt_model(args_model)


def model_getter_lookup(model_name: str) -> Callable:

    lookup_model = {'basic_fc': get_FC_model,
                    'highway': get_highway_model,
                    'pfn': get_pfn_model,
                    'efn': get_efn_model,
                    'transformer': get_transformer_model,
                    'part': get_part_model,
                    'depart': get_depart_model,
                    'bdt': get_bdt_model, }

    if model_name not in lookup_model:
        raise ValueError(f'Unknown model {model_name}')

    return lookup_model[model_name]
