import tensorflow as tf
from typing import Callable, Union, Tuple, Optional


class FFN(tf.keras.layers.Layer):
    def __init__(self, dim: int, expansion: int, activation: Callable, dropout: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim, self.expansion, self.activation, self.dropout = dim, expansion, activation, dropout

        self.wide_dense = tf.keras.layers.Dense(dim * expansion, activation=activation)
        self.dense = tf.keras.layers.Dense(dim, activation=None)
        self.ln = tf.keras.layers.LayerNormalization()
        self.layer_dropout = tf.keras.layers.Dropout(dropout)

    def get_config(self):
        config = super(FFN, self).get_config()
        config.update({"dim": self.dim, "expansion": self.expansion,
                      "activation": self.activation, "dropout": self.dropout})
        return config

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        output = self.wide_dense(inputs)
        output = self.ln(output)
        output = self.dense(output)
        if self.dropout > 0 and self.dropout is not None:
            output = self.layer_dropout(output)
        return output


# class MultiHeadSelfAttentionOLD(tf.keras.layers.Layer):
#     def __init__(self, dim: int, heads: int, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.dim, self.heads = dim, heads

#         for name in ["W_Q", "W_K", "W_V", "W_O"]:
#             setattr(self, name, self.add_weight(name=name, shape=[dim, dim]))

#     def get_config(self):
#         config = super(MultiheadSelfAttention, self).get_config()
#         config.update({"dim": self.dim, "heads": self.heads})
#         return config

#     def call(self, inputs: tf.Tensor, mask: Optional[tf.Tensor] = None, interaction: Optional[tf.Tensor] = None) -> tf.Tensor:

#         Q, K, V = [
#             tf.transpose(
#                 tf.reshape(
#                     inputs @ weights,
#                     [tf.shape(inputs)[0], tf.shape(inputs)[1], self.heads, self.dim // self.heads]),
#                 [0, 2, 1, 3])
#             for weights in [self.W_Q, self.W_K, self.W_V]
#         ]

#         attention_weights = tf.linalg.matmul(Q, K, transpose_b=True) / (Q.shape[-1] ** 0.5)

#         if interaction is not None:
#             # interaction is of shape [batch_size, N, N, heads]
#             interaction = tf.transpose(interaction, [0, 3, 1, 2])
#             attention_weights += interaction

#         attention = tf.keras.layers.Softmax()(attention_weights, mask=mask)

#         hidden = attention @ V
#         hidden = tf.transpose(hidden, [0, 2, 1, 3])
#         hidden = tf.reshape(hidden, [tf.shape(hidden)[0], tf.shape(hidden)[1], self.dim])
#         hidden = hidden @ self.W_O
#         return hidden


class MultiheadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, dim: int, heads: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim, self.heads = dim, heads

        self.linear_qkv = tf.keras.layers.Dense(dim * 3)
        self.linear_out = tf.keras.layers.Dense(dim)

    def get_config(self):
        config = super(MultiheadSelfAttention, self).get_config()
        config.update({"dim": self.dim, "heads": self.heads})
        return config

    def call(self, inputs: tf.Tensor, mask: Optional[tf.Tensor] = None, interaction: Optional[tf.Tensor] = None) -> tf.Tensor:
        B, N, C = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]

        qkv = self.linear_qkv(inputs)  # (B, N, C * 3)
        qkv = tf.reshape(qkv, [B, N, 3, self.heads, C // self.heads])  # (B, N, 3, H, C // H)
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])  # (3, B, H, N, C // H)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 3 x (B, H, N, C // H)

        attention_weights = tf.linalg.matmul(q, k, transpose_b=True) / (q.shape[-1] ** 0.5)  # (B, H, N, N)

        if interaction is not None:
            interaction = tf.transpose(interaction, [0, 3, 1, 2])  # (B, H, N, N)
            attention_weights += interaction

        attention = tf.keras.layers.Softmax()(attention_weights, mask=mask)  # (B, H, N, N)

        output = tf.linalg.matmul(attention, v)  # (B, H, N, C // H)
        output = tf.transpose(output, [0, 2, 1, 3])  # (B, N, H, C // H)
        output = tf.reshape(output, [B, N, C])  # (B, N, C)
        output = self.linear_out(output)  # (B, N, C)
        return output


class MultiheadClassAttention(tf.keras.layers.Layer):
    def __init__(self, dim: int, heads: int, dropout: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim, self.heads, self.dropout = dim, heads, dropout
        self.mha = tf.keras.layers.MultiHeadAttention(key_dim=dim // heads, num_heads=heads)
        self.layer_dropout = tf.keras.layers.Dropout(dropout)

    def get_config(self):
        config = super(MultiheadClassAttention, self).get_config()
        config.update({"dim": self.dim, "heads": self.heads, "dropout": self.dropout})
        return config

    def call(self, inputs: tf.Tensor, class_token: tf.Tensor, mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        output = self.mha(query=class_token, value=inputs, key=inputs, attention_mask=mask)
        if self.dropout is not None and self.dropout > 0:
            output = self.layer_dropout(output)
        return output


class SelfAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, dim: int, heads: int, dropout: float, expansion: int, activation: Callable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim, self.heads, self.dropout, self.expansion, self.activation = dim, heads, dropout, expansion, activation

        self.pre_mhsa_ln = tf.keras.layers.LayerNormalization()
        self.mhsa = MultiheadSelfAttention(dim=dim, heads=heads)
        self.post_mhsa_ln = tf.keras.layers.LayerNormalization()
        self.mhsa_dropout = tf.keras.layers.Dropout(dropout)

        self.pre_ffn_ln = tf.keras.layers.LayerNormalization()
        self.ffn = FFN(dim=dim, expansion=expansion, activation=activation, dropout=dropout)

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim, "heads": self.heads, "dropout": self.dropout,
                      "expansion": self.expansion, "activation": self.activation})
        return config

    def call(self, inputs: tf.Tensor, mask: Optional[tf.Tensor] = None, interaction: Optional[tf.Tensor] = None) -> tf.Tensor:
        attented = self.pre_mhsa_ln(inputs)
        attented = self.mhsa(inputs=attented, mask=mask, interaction=interaction)
        attented = self.post_mhsa_ln(attented)
        if self.dropout is not None and self.dropout > 0:
            attented = self.mhsa_dropout(attented)
        attented = attented + inputs

        ffned = self.pre_ffn_ln(attented)
        ffned = self.ffn(ffned)
        output = ffned + attented

        return output


class ClassAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, dim: int, heads: int, dropout: float, expansion: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim, self.heads, self.dropout, self.expansion = dim, heads, dropout, expansion

        self.pre_mhca_ln = tf.keras.layers.LayerNormalization()
        self.mhca = MultiheadClassAttention(dim=dim, heads=heads, dropout=dropout)
        self.post_mhca_ln = tf.keras.layers.LayerNormalization()
        self.mhca_dropout = tf.keras.layers.Dropout(dropout)

        self.pre_ffn_ln = tf.keras.layers.LayerNormalization()
        self.ffn = FFN(dim=dim, expansion=expansion, activation=tf.nn.gelu, dropout=dropout)

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim, "heads": self.heads, "dropout": self.dropout})
        return config

    def call(self, inputs: tf.Tensor, class_token: tf.Tensor, mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        attented = tf.concat([class_token, inputs], axis=1)
        attented = self.pre_mhca_ln(attented)
        attented = self.mhca(inputs=attented, class_token=class_token, mask=mask)
        attented = self.post_mhca_ln(attented)
        if self.dropout is not None and self.dropout > 0:
            attented = self.mhca_dropout(attented)
        attented = attented + class_token

        ffned = self.pre_ffn_ln(attented)
        ffned = self.ffn(ffned)
        output = ffned + attented
        return output


class ParT(tf.keras.layers.Layer):

    def __init__(self, dim: int, num_selfattn_layers: int, num_class_layers: int, expansion: int, heads: int, dropout: float, activation: Callable, *args, **kwargs):
        # Make sure `dim` is even.
        assert dim % 2 == 0

        super().__init__(*args, **kwargs)
        self.dim, self.expansion, self.heads, self.dropout, self.activation, self.num_particle_layers, self.num_class_layers = dim, expansion, heads, dropout, activation, num_selfattn_layers, num_class_layers

        self.class_token = tf.Variable(tf.random.truncated_normal((1, 1, dim), stddev=0.02), trainable=True)
        self.sa_layers = [SelfAttentionBlock(dim, heads, dropout, expansion, activation)
                          for _ in range(num_selfattn_layers)]
        self.ca_layers = [ClassAttentionBlock(dim, heads, dropout, expansion) for _ in range(num_class_layers)]

    def get_config(self):
        config = super(ParT, self).get_config()
        config.update({name: getattr(self, name)
                      for name in ["dim", "expansion", "heads", "dropout", "activation", "num_selfattn_layers", "num_class_layers"]})
        return config

    def call(self, inputs: tf.Tensor, mask: Optional[tf.Tensor] = None, interaction: Optional[tf.Tensor] = None) -> tf.Tensor:
        sa_mask = mask[:, tf.newaxis, tf.newaxis, :] & mask[:, tf.newaxis, :, tf.newaxis]
        hidden = inputs
        for layer in self.sa_layers:
            hidden = layer(hidden, sa_mask, interaction)

        class_token = tf.tile(self.class_token, [tf.shape(inputs)[0], 1, 1])
        class_mask = mask[:, tf.newaxis, :]
        class_mask = tf.concat([tf.ones((tf.shape(inputs)[0], 1, 1), dtype=tf.bool), class_mask], axis=2)
        for layer in self.ca_layers:
            class_token = layer(hidden, class_token, class_mask)
        return class_token


class Embedding(tf.keras.layers.Layer):

    def __init__(self, embedding_dim: int, num_embeding_layers: int, activation: Callable, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.embedding_dim, self.activation, self.num_embeding_layers = embedding_dim, activation, num_embeding_layers
        self.mlp = [tf.keras.layers.Dense(self.embedding_dim, activation=self.activation)
                    for _ in range(self.num_embeding_layers)]

    def get_config(self):
        config = super(Embedding, self).get_config()
        config.update({name: getattr(self, name) for name in ["embedding_dim", "num_embeding_layers", "activation"]})
        return config

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        hidden = inputs
        for layer in self.mlp:
            hidden = layer(hidden)
        return hidden


class CNNEmbedding(tf.keras.layers.Layer):

    def __init__(self, num_layers, layer_size, out_dim, activation, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation, self.num_layers, self.layer_size, self.out_dim = activation, num_layers, layer_size, out_dim

        self.conv_layers = [tf.keras.layers.Conv1D(layer_size, 1) for _ in range(num_layers)]
        self.conv_layers.append(tf.keras.layers.Conv1D(out_dim, 1))
        self.bn = [tf.keras.layers.BatchNormalization() for _ in range(num_layers + 1)]
        self.activation = tf.keras.layers.Activation(activation)

    def get_config(self):
        config = super(CNNEmbedding, self).get_config()
        config.update({name: getattr(self, name)
                      for name in ["num_layers", "layer_size", "out_dim", "activation"]})
        return config

    def call(self, inputs) -> tf.Tensor:
        # input shape (B, N, N, feature_dim)
        ones = tf.ones_like(inputs[0, :, :, 0])
        upper_tril_mask = tf.linalg.band_part(ones, 0, -1)
        diag_mask = tf.linalg.band_part(ones, 0, 0)
        upper_tril_mask = tf.cast(upper_tril_mask - diag_mask, tf.bool)
        flattened_upper_triag = tf.boolean_mask(inputs, upper_tril_mask, axis=1)

        hidden = flattened_upper_triag
        for conv, norm in zip(self.conv_layers, self.bn):
            hidden = conv(hidden)
            hidden = norm(hidden)
            hidden = self.activation(hidden)

        true_mask = tf.cast(tf.where(upper_tril_mask), tf.int32)
        out = tf.transpose(hidden, [1, 0, 2])
        out = tf.scatter_nd(true_mask, out, shape=[tf.shape(inputs)[1],
                            tf.shape(inputs)[2], tf.shape(inputs)[0], self.out_dim])
        out = out + tf.transpose(out, [1, 0, 2, 3])
        out = tf.transpose(out, [2, 0, 1, 3])
        return out


class ParTModel(tf.keras.Model):

    def __init__(self,
                 input_shape: Union[Tuple[int], Tuple[Tuple[int], Tuple[int]]],
                 embedding_dim: int,
                 num_embeding_layers: int,
                 selfattn_block_layers: int,
                 class_block_layers: int,
                 expansion: int,
                 heads: int,
                 dropout: float,
                 output_layer: tf.keras.layers.Layer,
                 activation: Callable[[tf.Tensor], tf.Tensor],
                 interaction_embedding_num_layers: Optional[int] = None,
                 interaction_embedding_layer_size: Optional[int] = None,
                 preprocess: Union[tf.keras.layers.Layer, None, Tuple[tf.keras.layers.Layer, tf.keras.layers.Layer]] = None):

        if isinstance(input_shape, tuple) and isinstance(input_shape[0], tuple):
            input = (tf.keras.layers.Input(shape=input_shape[0], ragged=True),
                     tf.keras.layers.Input(shape=input_shape[1], ragged=True))
            row_lengths = input[0].row_lengths()
            hidden = input[0].to_tensor()
            interaction_hidden = input[1].to_tensor()

            if preprocess is not None:
                preprocess, interaction_preprocess = preprocess
                if interaction_preprocess is not None:
                    interaction_hidden = interaction_preprocess(interaction_hidden)

            embed_interaction = CNNEmbedding(
                interaction_embedding_num_layers,
                interaction_embedding_layer_size,
                heads,
                activation)(interaction_hidden)
        else:
            input = tf.keras.layers.Input(shape=input_shape, ragged=True)
            embed_interaction = None
            row_lengths = input.row_lengths()
            hidden = input.to_tensor()

        if preprocess is not None:
            hidden = preprocess(hidden)

        hidden = Embedding(embedding_dim, num_embeding_layers, activation)(hidden)

        transformed = ParT(dim=embedding_dim,
                           num_selfattn_layers=selfattn_block_layers,
                           num_class_layers=class_block_layers,
                           expansion=expansion,
                           heads=heads,
                           dropout=dropout,
                           activation=activation)(hidden, tf.sequence_mask(row_lengths), embed_interaction)

        transformed = tf.keras.layers.LayerNormalization()(transformed)
        output = output_layer(transformed[:, 0, :])

        super().__init__(inputs=input, outputs=output)
