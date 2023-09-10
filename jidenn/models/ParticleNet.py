import tensorflow as tf
from typing import Callable, Union, Tuple, Optional, Literal, List


# class MatrixDistance(tf.keras.layers.Layer):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def call(self, A, B):
#         # A shape is (N, P_A, C), B shape is (N, P_B, C)
#         # D shape is (N, P_A, P_B)
#         r_A = tf.reduce_sum(A * A, axis=2, keepdims=True)
#         r_B = tf.reduce_sum(B * B, axis=2, keepdims=True)
#         m = tf.matmul(A, tf.transpose(B, perm=(0, 2, 1)))
#         D = r_A - 2 * m + tf.transpose(r_B, perm=(0, 2, 1))
#         return D


class kNN(tf.keras.layers.Layer):
    def __init__(self, num_points: int, k: int):
        super().__init__()
        self.num_points = num_points
        self.k = k
        self.top_k = tf.keras.layers.Lambda(lambda x: tf.nn.top_k(x, k=self.k + 1))

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


class EdgeConv(tf.keras.layers.Layer):

    def __init__(self, k: int, channels: List[int], num_points: int, pooling: Literal['average', 'max'], activation: Callable[[tf.Tensor], tf.Tensor]):
        super().__init__()
        self.k = k
        self.activation = activation
        self.pooling = pooling
        self.channels = channels
        self.num_points = num_points

        self.knn_layer = kNN(num_points, k)
        self.layers = [tf.keras.layers.Dense(channel, use_bias=False, activation=None) for channel in channels]
        self.bn_layers = [tf.keras.layers.BatchNormalization() for _ in channels]

        self.bypass = tf.keras.layers.Dense(channels[-1], use_bias=False, activation=None)
        self.bypass_bn = tf.keras.layers.BatchNormalization()

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


class ParticleNet(tf.keras.layers.Layer):

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

        self.bn = tf.keras.layers.BatchNormalization()
        self.edge_convs = [EdgeConv(k_neighbours, channels, num_points, pooling, activation)
                           for k_neighbours, channels in zip(edge_knn, edge_layers)]
        self.fc_layers = [tf.keras.layers.Dense(units, activation=activation) for units in fc_layers]
        self.dropout_layers = [tf.keras.layers.Dropout(dropout) for dropout in fc_dropout]

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
        mask = tf.cast(tf.not_equal(mask, 0), dtype=tf.float32)  # 1 if valid
        coord_shift = tf.multiply(999., tf.cast(tf.equal(mask, 0), dtype=tf.float32))  # make non-valid positions to 99

        features = self.bn(features)

        for layer_idx, layer in enumerate(self.edge_convs):
            points = coord_shift + points if layer_idx == 0 else coord_shift + features

            features = layer(points, features)

        features = tf.multiply(features, mask)
        hidden = tf.keras.layers.GlobalAveragePooling1D()(features)

        for layer, dropout in zip(self.fc_layers, self.dropout_layers):
            hidden = layer(hidden)
            hidden = dropout(hidden)

        return hidden


class ParticleNetModel(tf.keras.Model):

    def __init__(self,
                 input_shape: Tuple[Tuple[int, int], Tuple[int, int]],
                 output_layer: tf.keras.layers.Layer,
                 activation: Callable[[tf.Tensor], tf.Tensor],
                 preprocess: Union[Tuple[tf.keras.layers.Layer, tf.keras.layers.Layer], None] = None,
                 pooling: Literal['average', 'max'] = 'average',
                 fc_layers: List[int] = [256],
                 fc_dropout: List[float] = [0.1],
                 edge_knn: List[int] = [16, 16, 16],
                 edge_layers: List[List[int]] = [[64, 64, 64], [128, 128, 128], [256, 256, 256]],
                 ):

        input = (tf.keras.Input(name='points', shape=input_shape[0]),
                 tf.keras.Input(name='features', shape=input_shape[1]),
                 tf.keras.Input(name='mask', shape=input_shape[2]))

        points = input[0]
        features = input[1]
        mask = input[2]

        if preprocess is not None:
            points = preprocess[0](points)
            features = preprocess[1](features)

        # row_lengths = points.row_lengths()
        # mask = tf.sequence_mask(row_lengths)

        # points = points.to_tensor()
        # features = features.to_tensor()

        num_points = 100
        # pad points and features shape to (N, 100, C)
        # points = tf.pad(points, [[0, 0], [0, num_points - tf.shape(points)[1]], [0, 0]])
        # features = tf.pad(features, [[0, 0], [0, num_points - tf.shape(features)[1]], [0, 0]])
        # mask = tf.pad(mask, [[0, 0], [0, num_points - tf.shape(mask)[1]]])

        output = ParticleNet(
            pooling=pooling,
            num_points=num_points,
            fc_layers=fc_layers,
            fc_dropout=fc_dropout,
            edge_knn=edge_knn,
            edge_layers=edge_layers,
            activation=activation,
        )(points=points, features=features, mask=mask)

        output = output_layer(output)

        super().__init__(inputs=input, outputs=output)
