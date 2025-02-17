import tensorflow as tf
import keras
from typing import Callable, Union, Tuple, Literal, List

class kNN(keras.layers.Layer):
    def __init__(self, num_points: int, k: int):
        super().__init__()
        self.num_points = num_points
        self.k = k
        self.top_k = keras.layers.Lambda(lambda x: tf.nn.top_k(x, k=self.k + 1))

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_points': self.num_points,
            'k': self.k,
        })
        return config

    def call(self, features, points):
        distance = tf.norm(tf.expand_dims(points, axis=2) - tf.expand_dims(points, axis=1), axis=-1)
        _, topk_indices = self.top_k(-distance)  # (N, P, K+1)
        topk_indices = topk_indices[:, :, 1:]  # (N, P, K)

        # batch_size = tf.shape(features)[0]
        # batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, self.num_points, self.k, 1))
        # indices = tf.concat([batch_indices, tf.expand_dims(topk_indices, axis=3)], axis=3)  # (N, P, K, 2)
        batch_size = tf.shape(features)[0]
        batch_indices = tf.range(batch_size, dtype=tf.int32)
        batch_indices = tf.reshape(batch_indices, (batch_size, 1, 1, 1))
        batch_indices = tf.broadcast_to(batch_indices, (batch_size, self.num_points, self.k, 1))
        indices = tf.concat([batch_indices, tf.expand_dims(topk_indices, axis=3)], axis=3)  # (N, P, K, 2)
        return indices


class EdgeConv(keras.layers.Layer):

    def __init__(self, k: int, channels: List[int], num_points: int, pooling: Literal['average', 'max'], activation: Callable[[tf.Tensor], tf.Tensor]):
        super().__init__()
        self.k = k
        self.activation = activation
        self.pooling = pooling
        self.channels = channels
        self.num_points = num_points

        self.knn_layer = kNN(num_points, k)
        self.layers = [keras.layers.Dense(channel, use_bias=False, activation=None) for channel in channels]
        self.bn_layers = [keras.layers.BatchNormalization() for _ in channels]

        self.bypass = keras.layers.Dense(channels[-1], use_bias=False, activation=None)
        self.bypass_bn = keras.layers.BatchNormalization()

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'k': self.k,
            'activation': self.activation,
            'pooling': self.pooling,
            'channels': self.channels,
            'num_points': self.num_points,
        })
        return config

    def call(self, points, features):
        knn_indices = self.knn_layer(features, points)
        knn_fts = tf.gather_nd(features, knn_indices)
        # knn_fts_center = tf.tile(tf.expand_dims(features, axis=2), (1, 1, self.k, 1))  # (N, P, K, C)
        # hidden = tf.concat([knn_fts_center, tf.subtract(knn_fts, knn_fts_center)], axis=-1)  # (N, P, K, 2*C)
        knn_fts_center = tf.expand_dims(features, axis=2)  # (N, P, 1, C)
        knn_fts_center = tf.broadcast_to(knn_fts_center, (tf.shape(features)[0], self.num_points, self.k, tf.shape(features)[-1]))  # (N, P, K, C)
        hidden = tf.concat([knn_fts_center, knn_fts - knn_fts_center], axis=-1)  # (N, P, K, 2*C)

        for layer, bn in zip(self.layers, self.bn_layers):
            hidden = layer(hidden)
            hidden = bn(hidden)
            hidden = self.activation(hidden)

        if self.pooling == 'max':
            hidden = tf.reduce_max(hidden, axis=2)  # (N, P, C')
        elif self.pooling == 'average':
            hidden = tf.reduce_mean(hidden, axis=2)  # (N, P, C')
        else:
            raise ValueError(f'Pooling method {self.pooling} not implemented. Use "max" or "average"')

        features = self.bypass(features)
        features = self.bypass_bn(features)

        output = features + hidden
        output = self.activation(output)
        return output


class ParticleNet(keras.layers.Layer):

    def __init__(self,
                 num_points: int,
                 activation: Callable[[tf.Tensor], tf.Tensor],
                 pooling: Literal['average', 'max'] = 'average',
                 fc_layers: List[int] = [256],
                 fc_dropout: List[float] = [0.1],
                 edge_knn: List[int] = [16, 16, 16],
                 edge_layers: List[List[int]] = [[64, 64, 64], [128, 128, 128], [256, 256, 256]],
                 ):
        super().__init__()
        assert len(fc_layers) == len(fc_dropout), 'fc_layers and fc_dropout must have the same length'
        assert len(edge_knn) == len(
            edge_layers), 'edge_conv_knn and edge_conv_channels must have the same length'

        self.num_points = num_points
        self.activation = activation
        self.pooling = pooling
        self.fc_layers = fc_layers
        self.fc_dropout = fc_dropout
        self.edge_knn = edge_knn
        self.edge_layers = edge_layers

        self.bn = keras.layers.BatchNormalization()
        self.edge_convs = [EdgeConv(k_neighbours, channels, num_points, pooling, activation)
                           for k_neighbours, channels in zip(edge_knn, edge_layers)]
        self.fc_layers = [keras.layers.Dense(units, activation=activation) for units in fc_layers]
        self.dropout_layers = [keras.layers.Dropout(dropout) for dropout in fc_dropout]

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_points': self.num_points,
            'activation': self.activation,
            'pooling': self.pooling,
            'fc_layers': self.fc_layers,
            'fc_dropout': self.fc_dropout,
            'edge_knn': self.edge_knn,
            'edge_layers': self.edge_layers,
        })
        return config

    def call(self, points, features, mask):
        mask = tf.expand_dims(mask, axis=-1)
        mask = tf.cast(tf.not_equal(mask, False), dtype=tf.float32)  # 1 if valid
        coord_shift = tf.multiply(999., tf.cast(tf.equal(mask, 0), dtype=tf.float32))  # make non-valid positions to 99

        features = self.bn(features)

        for layer_idx, layer in enumerate(self.edge_convs):
            points = coord_shift + points if layer_idx == 0 else coord_shift + features

            features = layer(points, features)

        features = tf.multiply(features, mask)
        hidden = keras.layers.GlobalAveragePooling1D()(features)

        for layer, dropout in zip(self.fc_layers, self.dropout_layers):
            hidden = layer(hidden)
            hidden = dropout(hidden)

        return hidden


class ParticleNetModel(keras.Model):

    def __init__(self,
                 input_shape: Tuple[Tuple[int, int], Tuple[int, int]],
                 output_layer: keras.layers.Layer,
                 activation: Callable[[tf.Tensor], tf.Tensor],
                 preprocess: Union[Tuple[keras.layers.Layer, keras.layers.Layer], None] = None,
                 pooling: Literal['average', 'max'] = 'average',
                 fc_layers: List[int] = [256],
                 fc_dropout: List[float] = [0.1],
                 edge_knn: List[int] = [16, 16, 16],
                 edge_layers: List[List[int]] = [[64, 64, 64], [128, 128, 128], [256, 256, 256]],
                 max_constituents: int = 100,
                 **kwargs
                 ):
        
        self.input_size, self.output_layer, self.preprocess = input_shape, output_layer, preprocess
        self.pooling = pooling
        self.fc_layers = fc_layers
        self.fc_dropout = fc_dropout
        self.edge_knn = edge_knn
        self.edge_layers = edge_layers
        self.activation = activation
        self.max_constituents = max_constituents
        
        

        input = (keras.Input(shape=input_shape[0]),
                 keras.Input(shape=input_shape[1]),
                 keras.layers.Input(shape=(input_shape[0][0],), dtype=tf.bool))

        features = input[0]
        points = input[1]
        mask = input[2]

        if preprocess is not None:
            features = preprocess[0](features)
            points = preprocess[1](points)


        output = ParticleNet(
            pooling=pooling,
            num_points=max_constituents,
            fc_layers=fc_layers,
            fc_dropout=fc_dropout,
            edge_knn=edge_knn,
            edge_layers=edge_layers,
            activation=activation,
        )(points=points, features=features, mask=mask)

        output = output_layer(output)

        super().__init__(inputs=input, outputs=output, **kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'input_shape': self.input_size,
            'output_layer': keras.layers.serialize(self.output_layer),
            'preprocess': keras.layers.serialize(self.preprocess),
            'pooling': self.pooling,
            'fc_layers': self.fc_layers,
            'fc_dropout': self.fc_dropout,
            'edge_knn': self.edge_knn,
            'edge_layers': self.edge_layers,
            'activation': keras.activations.serialize(self.activation),
            'max_constituents': self.max_constituents,
        })
        return config
    
    @classmethod   
    def from_config(cls, config):
        config['output_layer'] = keras.layers.deserialize(config['output_layer'])
        config['preprocess'] = keras.layers.deserialize(config['preprocess'])
        config['activation'] = keras.activations.deserialize(config['activation'])
        return cls(**config)