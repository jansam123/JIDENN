import tensorflow as tf
import tensorflow_addons as tfa
from typing import Union, Tuple, Optional, Callable, List

from src.config import config_subclasses as cfg
from src.config import model_config as model_cfg
from .get_optimizer import get_optimizer

from .BasicFCModel import BasicFCModel
from .HighwayModel import HighwayModel
from .TransformerModel import TransformerModel
from .ParTModel import ParTModel
from .DeParTModel import DeParTModel
from .BDT import get_BDT_model

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

    return BasicFCModel(
        hidden_layer_size=args_model.layer_size,
        num_layers=args_model.num_layers,
        dropout=args_model.dropout,
        input_size=input_size,
        output_layer=output_layer,
        activation=get_activation(args_model.activation),
        preprocess=preprocess)

def get_highway_model(input_size,
                 output_layer,
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

def get_transformer_model(input_size,
                          output_layer,
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
                   output_layer,
                   args_model: model_cfg.ParT,
                   preprocess: Optional[tf.keras.layers.Layer] = None):
    
    return ParTModel(
        input_shape=input_size,
        output_layer=output_layer,
        #
        embedding_dim=args_model.embed_dim,
        num_embeding_layers=args_model.num_embed_layers,
        particle_block_layers=args_model.particle_block_layers,
        class_block_layers=args_model.class_block_layers,
        expansion=args_model.expansion,
        heads=args_model.heads,
        particle_block_dropout=args_model.particle_block_dropout,
        interaction=args_model.interaction,
        interaction_embedding_num_layers=args_model.interaction_embedding_num_layers,
        interaction_embedding_layer_size=args_model.interaction_embedding_layer_size,
        #
        preprocess=preprocess,
        activation=get_activation(args_model.activation))

def get_depart_model(input_size,
                     output_layer,
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
        interaction=args_model.interaction,
        layer_scale_init_value=args_model.layer_scale_init_value,
        stochastic_depth_drop_rate=args_model.stochastic_depth_drop_rate,
        #
        preprocess=preprocess,
        activation=get_activation(args_model.activation))
    

def get_compiled_model(model_name: str, 
                       input_size: Union[int, Tuple[int]],
                       args_models: cfg.Models,  
                       args_optimizer: cfg.Optimizer,
                       num_labels: int,
                       preprocess: Union[tf.keras.layers.Layer, None] = None):
    
    if model_name == 'basic_fc':
        model = get_FC_model(input_size,
                             get_output_layer(num_labels),
                             args_models.basic_fc,
                             preprocess)
        
    elif model_name == 'highway':
        model = get_highway_model(input_size,
                                  get_output_layer(num_labels),
                                  args_models.highway,
                                  preprocess)
    elif model_name == 'transformer':
        model = get_transformer_model(input_size,
                                      get_output_layer(num_labels),
                                      args_models.transformer,
                                      preprocess)
    elif model_name == 'part':
        model = get_part_model(input_size,
                               get_output_layer(num_labels),
                               args_models.part,
                               preprocess)
    elif model_name == 'depart':
        model = get_depart_model(input_size,
                                 get_output_layer(num_labels),
                                 args_models.depart,
                                 preprocess)
    elif model_name == 'bdt':
        model = get_BDT_model(args_models.bdt)    
        model.compile(weighted_metrics=get_metrics(num_labels))
        return model
    
    else:
        raise NotImplementedError(f'Model {model_name} not supported.')
    
    model.compile(optimizer=get_optimizer(args_optimizer),
                  loss=get_loss(num_labels, args_optimizer.label_smoothing),
                  weighted_metrics=get_metrics(num_labels))

    return model
                         
    
    
    
        



