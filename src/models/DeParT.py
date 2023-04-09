
import tensorflow as tf
from typing import Callable, Union, Tuple, Optional


class FFN(tf.keras.layers.Layer):
    def __init__(self, dim: int, expansion: int, activation: Callable, dropout: Optional[float] = None, **kwargs):
        super().__init__(**kwargs)
        self.dim, self.expansion, self.activation, self.dropout = dim, expansion, activation, dropout

        self.wide_dense = tf.keras.layers.Dense(int(dim * expansion * 2 / 3), activation=tf.nn.relu, use_bias=False)
        self.gate_dense = tf.keras.layers.Dense(int(dim * expansion * 2 / 3), activation=None, use_bias=False)
        self.dense = tf.keras.layers.Dense(dim, activation=None, use_bias=False)
        self.layer_dropout = tf.keras.layers.Dropout(dropout)

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim, "expansion": self.expansion,
                      "activation": self.activation, "dropout": self.dropout})
        return config

    def call(self, inputs):
        output = self.wide_dense(inputs) * self.gate_dense(inputs)
        output = self.dense(output)
        output = self.layer_dropout(output)
        return output


class LayerScale(tf.keras.layers.Layer):

    def __init__(self, init_values: float, projection_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.gamma = tf.Variable(init_values * tf.ones((projection_dim,)))

    def call(self, x, training=False):
        return x * self.gamma


class StochasticDepth(tf.keras.layers.Layer):

    def __init__(self, drop_prob: float, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x, training=False):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x


# class TalkingHeadAttention(tf.keras.layers.Layer):

#     def __init__(self, dim: int, heads: int, dropout: float, **kwargs):
#         super().__init__(**kwargs)

#         self.num_heads = heads
#         head_dim = dim // self.num_heads

#         self.scale = head_dim**-0.5

#         self.qkv = tf.keras.layers.Dense(dim * 3)
#         self.attn_drop = tf.keras.layers.Dropout(dropout)

#         self.proj = tf.keras.layers.Dense(dim)

#         self.proj_l = tf.keras.layers.Dense(self.num_heads)
#         self.proj_w = tf.keras.layers.Dense(self.num_heads)

#         self.proj_drop = tf.keras.layers.Dropout(dropout)

#     def call(self, x, mask, interaction=None, training=False):
#         B, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]

#         # Project the inputs all at once.
#         qkv = self.qkv(x)

#         # Reshape the projected output so that they're segregated in terms of
#         # query, key, and value projections.
#         qkv = tf.reshape(qkv, (B, N, 3, self.num_heads, C // self.num_heads))

#         # Transpose so that the `num_heads` becomes the leading dimensions.
#         # Helps to better segregate the representation sub-spaces.
#         qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])  # 3, B, num_heads, N, C // num_heads
#         scale = tf.cast(self.scale, dtype=qkv.dtype)
#         q, k, v = qkv[0] * scale, qkv[1], qkv[2]  # B, num_heads, N, C // num_heads

#         # Permute the key to match the shape of the query.
#         k = tf.transpose(k, perm=[0, 1, 3, 2])  # B, num_heads, C // num_heads, N
#         # Obtain the raw attention scores.
#         attn = tf.matmul(q, k)  # B, num_heads, N, N

#         # Linear projection of the similarities between the query and key projections.
#         attn = self.proj_l(tf.transpose(attn, perm=[0, 2, 3, 1]))

#         # Normalize the attention scores.
#         attn = tf.transpose(attn, perm=[0, 3, 1, 2])

#         if interaction is not None:
#             interaction = tf.transpose(interaction, perm=[0, 3, 1, 2])
#             attn += interaction

#         attn = tf.keras.layers.Softmax()(attn, mask=mask)

#         # Linear projection on the softmaxed scores.
#         attn = self.proj_w(tf.transpose(attn, perm=[0, 2, 3, 1]))
#         attn = tf.transpose(attn, perm=[0, 3, 1, 2])
#         attn = self.attn_drop(attn, training)

#         # Final set of projections as done in the vanilla attention mechanism.
#         x = tf.matmul(attn, v)
#         x = tf.transpose(x, perm=[0, 2, 1, 3])
#         x = tf.reshape(x, (B, N, C))

#         x = self.proj(x)
#         x = self.proj_drop(x, training)

#         return x


class TalkingMultiheadSelfAttention(tf.keras.layers.Layer):

    def __init__(self, dim, heads, dropout, * args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim, self.heads = dim, heads

        self.linear_qkv = tf.keras.layers.Dense(dim * 3)
        self.linear_out = tf.keras.layers.Dense(dim)

        self.linear_talking_1 = tf.keras.layers.Dense(heads)
        self.linear_talking_2 = tf.keras.layers.Dense(heads)

        self.dropout = tf.keras.layers.Dropout(dropout)
        self.attn_drop = tf.keras.layers.Dropout(dropout)

    def get_config(self):
        config = super(TalkingMultiheadSelfAttention, self).get_config()
        config.update({"dim": self.dim, "heads": self.heads})
        return config

    def call(self, inputs, mask, interaction=None, training=False):
        B, N, C = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]

        qkv = self.linear_qkv(inputs)  # (B, N, C * 3)
        qkv = tf.reshape(qkv, [B, N, 3, self.heads, C // self.heads])  # (B, N, 3, H, C // H)
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])  # (3, B, H, N, C // H)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 3 x (B, H, N, C // H)

        attention_weights = tf.linalg.matmul(q, k, transpose_b=True) / (q.shape[-1] ** 0.5)  # (B, H, N, N)

        attention_weights = self.linear_talking_1(tf.transpose(attention_weights, [0, 2, 3, 1]))  # (B, N, N, H)
        attention_weights = tf.transpose(attention_weights, [0, 3, 1, 2])  # (B, H, N, N)

        if interaction is not None:
            interaction = tf.transpose(interaction, [0, 3, 1, 2])  # (B, H, N, N)
            attention_weights += interaction

        attention = tf.keras.layers.Softmax()(attention_weights, mask=mask)  # (B, H, N, N)
        attention = self.linear_talking_2(tf.transpose(attention, [0, 2, 3, 1]))  # (B, N, N, H)
        attention = tf.transpose(attention, [0, 3, 1, 2])  # (B, H, N, N)
        attention = self.attn_drop(attention, training)  # (B, H, N, N)

        output = tf.linalg.matmul(attention, v)  # (B, H, N, C // H)
        output = tf.transpose(output, [0, 2, 1, 3])  # (B, N, H, C // H)
        output = tf.reshape(output, [B, N, C])  # (B, N, C)
        output = self.linear_out(output, training)  # (B, N, C)
        return output


class TalkingMultiheadClassAttention(tf.keras.layers.Layer):

    def __init__(self, dim, heads, dropout, * args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim, self.heads = dim, heads

        self.linear_kv = tf.keras.layers.Dense(dim * 2)
        self.linear_q = tf.keras.layers.Dense(dim)
        self.linear_out = tf.keras.layers.Dense(dim)

        self.linear_talking_1 = tf.keras.layers.Dense(heads)
        self.linear_talking_2 = tf.keras.layers.Dense(heads)

        self.dropout = tf.keras.layers.Dropout(dropout)
        self.attn_drop = tf.keras.layers.Dropout(dropout)

    def get_config(self):
        config = super(TalkingMultiheadClassAttention, self).get_config()
        config.update({"dim": self.dim, "heads": self.heads})
        return config

    def call(self, inputs, class_token, mask, training=False):
        B, N, C = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]

        kv = self.linear_kv(inputs)  # (B, N, C * 3)
        kv = tf.reshape(kv, [B, N, 2, self.heads, C // self.heads])  # (B, N, 3, H, C // H)
        kv = tf.transpose(kv, [2, 0, 3, 1, 4])  # (3, B, H, N, C // H)
        k, v = kv[0], kv[1]  # 2 x (B, H, N, C // H)

        q = self.linear_q(class_token)  # (B, 1, C)
        q = tf.reshape(q, [B, self.heads, 1, C // self.heads])  # (B, H, 1, C // H)

        attention_weights = tf.linalg.matmul(q, k, transpose_b=True) / (q.shape[-1] ** 0.5)  # (B, H, 1, N)

        attention_weights = self.linear_talking_1(tf.transpose(attention_weights, [0, 2, 3, 1]))  # (B, 1, N, H)
        attention_weights = tf.transpose(attention_weights, [0, 3, 1, 2])  # (B, H, 1, N)

        attention = tf.keras.layers.Softmax()(attention_weights, mask=mask)  # (B, H, 1, N)
        attention = self.linear_talking_2(tf.transpose(attention, [0, 2, 3, 1]))  # (B, 1, N, H)
        attention = tf.transpose(attention, [0, 3, 1, 2])  # (B, H, 1, N)
        attention = self.attn_drop(attention, training)  # (B, H, 1, N)

        output = tf.linalg.matmul(attention, v)  # (B, H, 1, C // H)
        output = tf.transpose(output, [0, 2, 1, 3])  # (B, 1, H, C // H)
        output = tf.reshape(output, [B, 1, C])  # (B, 1, C)
        output = self.linear_out(output, training)  # (B, 1, C)
        return output


class MultiheadClassAttention(tf.keras.layers.Layer):
    def __init__(self, dim, heads, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dim, self.heads, self.dropout = dim, heads, dropout
        self.mha = tf.keras.layers.MultiHeadAttention(key_dim=dim // heads, num_heads=heads, dropout=dropout)

    def get_config(self):
        config = super(MultiheadClassAttention, self).get_config()
        config.update({"dim": self.dim, "heads": self.heads, "dropout": self.dropout})
        return config

    def call(self, query, inputs, mask):
        # Execute the Self-Attention Transformer layer.
        output = self.mha(query=query, value=inputs, key=inputs, attention_mask=mask)
        return output


class SelfAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, dim, heads, dropout, stoch_drop_prob, layer_scale_init_value, activation, expansion, **kwargs):
        super().__init__(**kwargs)
        self.dim, self.heads, self.dropout, self.stoch_drop_prob, self.layer_scale_init_value, self.activation, self.expansion = dim, heads, dropout, stoch_drop_prob, layer_scale_init_value, activation, expansion

        self.pre_mhsa_ln = tf.keras.layers.LayerNormalization()
        self.mhsa = TalkingMultiheadSelfAttention(dim, heads, dropout)
        self.post_mhsa_scale = LayerScale(layer_scale_init_value, dim)
        self.post_mhsa_stoch_depth = StochasticDepth(drop_prob=stoch_drop_prob)

        self.pre_ffn_ln = tf.keras.layers.LayerNormalization()
        self.ffn = FFN(dim, expansion, activation, dropout)
        self.post_ffn_scale = LayerScale(layer_scale_init_value, dim)
        self.post_ffn_stoch_depth = StochasticDepth(drop_prob=stoch_drop_prob)

    def get_config(self):
        config = super(SelfAttentionBlock, self).get_config()
        config.update({"dim": self.dim, "heads": self.heads, "dropout": self.dropout, "stoch_drop_prob": self.stoch_drop_prob,
                      "layer_scale_init_value": self.layer_scale_init_value, "activation": self.activation, "expansion": self.expansion})
        return config

    def call(self, inputs, mask, interaction=None):
        attented = self.pre_mhsa_ln(inputs)
        attented = self.mhsa(attented, mask, interaction)
        attented = self.post_mhsa_scale(attented)
        attented = self.post_mhsa_stoch_depth(attented)
        attented = attented + inputs

        ffned = self.pre_ffn_ln(attented)
        ffned = self.ffn(ffned)
        ffned = self.post_ffn_scale(ffned)
        ffned = self.post_ffn_stoch_depth(ffned)
        output = ffned + attented

        return output


class ClassAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, dim, heads, dropout, stoch_drop_prob, layer_scale_init_value, activation, expansion, **kwargs):
        super().__init__(**kwargs)
        self.dim, self.heads, self.dropout, self.stoch_drop_prob, self.layer_scale_init_value, self.activation, self.expansion = dim, heads, dropout, stoch_drop_prob, layer_scale_init_value, activation, expansion

        self.pre_mhca_ln = tf.keras.layers.LayerNormalization()
        self.mhca = MultiheadClassAttention(dim, heads, dropout)
        self.post_mhca_scale = LayerScale(layer_scale_init_value, dim)
        self.post_mhca_stoch_depth = StochasticDepth(drop_prob=stoch_drop_prob)

        self.pre_ffn_ln = tf.keras.layers.LayerNormalization()
        self.ffn = FFN(dim, expansion, activation, dropout)
        self.post_ffn_scale = LayerScale(layer_scale_init_value, dim)
        self.post_ffn_stoch_depth = StochasticDepth(drop_prob=stoch_drop_prob)

    def get_config(self):
        config = super(ClassAttentionBlock, self).get_config()
        config.update({"dim": self.dim, "heads": self.heads, "dropout": self.dropout, "stoch_drop_prob": self.stoch_drop_prob,
                      "layer_scale_init_value": self.layer_scale_init_value, "activation": self.activation, "expansion": self.expansion})
        return config

    def call(self, inputs, cls_token, mask):
        inputs = tf.concat([cls_token, inputs], axis=1)

        attented = self.pre_mhca_ln(inputs)
        attented = self.mhca(query=cls_token, inputs=attented, mask=mask)
        attented = self.post_mhca_scale(attented)
        attented = self.post_mhca_stoch_depth(attented)
        attented = attented + cls_token

        ffned = self.pre_ffn_ln(attented)
        ffned = self.ffn(ffned)
        ffned = self.post_ffn_scale(ffned)
        ffned = self.post_ffn_stoch_depth(ffned)
        output = ffned + attented
        return output


class DeParT(tf.keras.layers.Layer):
    def __init__(self,
                 num_selfattn_layers: int,
                 num_class_layers: int,
                 dim: int,
                 expansion: int,
                 heads: int,
                 dropout: float,
                 class_dropout: float,
                 activation: Callable,
                 layer_scale_init_value: float,
                 stochastic_depth_drop_rate: float,
                 class_stochastic_depth_drop_rate: float,
                 * args, **kwargs):
        # Make sure `dim` is even.
        assert dim % 2 == 0

        super().__init__(**kwargs)
        self.layers, self.dim, self.expansion, self.heads, self.dropout, self.activation, self.class_layers = num_selfattn_layers, dim, expansion, heads, dropout, activation, num_class_layers
        self.layer_scale_init_value, self.stochastic_depth_drop_rate, self.class_stochastic_depth_drop_rate = layer_scale_init_value, stochastic_depth_drop_rate, class_stochastic_depth_drop_rate
        self.class_dropout = class_dropout

        self.sa_layers = [SelfAttentionBlock(dim,
                                             heads,
                                             dropout,
                                             self.stochastic_prob(i, num_selfattn_layers - 1,
                                                                  stochastic_depth_drop_rate),
                                             layer_scale_init_value,
                                             activation,
                                             expansion) for i in range(num_selfattn_layers)]

        self.ca_layers = [ClassAttentionBlock(dim,
                                              heads,
                                              class_dropout,
                                              self.stochastic_prob(i, num_class_layers - 1,
                                                                   class_stochastic_depth_drop_rate),
                                              layer_scale_init_value,
                                              activation,
                                              expansion) for i in range(num_class_layers)]

        self.cls_token = tf.Variable(tf.random.truncated_normal((1, 1, dim), stddev=0.02), trainable=True)

    def stochastic_prob(self, step, total_steps, drop_rate):
        return drop_rate * step / total_steps

    def get_config(self):
        config = super(DeParT, self).get_config()
        config.update({name: getattr(self, name)
                      for name in ["layers", "dim", "expansion", "heads", "dropout", "activation", "class_layers",
                                   "layer_scale_init_value", "stochastic_depth_drop_rate", "class_stochastic_depth_drop_rate", "class_dropout"]})
        return config

    def call(self, inputs, mask, interaction=None):
        sa_mask = mask[:, tf.newaxis, tf.newaxis, :] & mask[:, tf.newaxis, :, tf.newaxis]
        hidden = inputs
        for layer in self.sa_layers:
            hidden = layer(hidden, sa_mask, interaction)

        cls_token = tf.tile(self.cls_token, (tf.shape(inputs)[0], 1, 1))
        class_mask = mask[:, tf.newaxis, :]
        class_mask = tf.concat([tf.ones((tf.shape(inputs)[0], 1, 1), dtype=tf.bool), class_mask], axis=2)
        for layer in self.ca_layers:
            cls_token = layer(hidden, cls_token=cls_token, mask=class_mask)
        return cls_token


class ParticleEmbedding(tf.keras.layers.Layer):

    def __init__(self, embedding_dim, num_embeding_layers, activation, **kwargs):

        super().__init__(**kwargs)
        self.embedding_dim, self.activation, self.num_embeding_layers = embedding_dim, activation, num_embeding_layers
        self.mlp = [tf.keras.layers.Dense(self.embedding_dim, activation=self.activation)
                    for _ in range(self.num_embeding_layers)]

    def get_config(self):
        config = super(ParticleEmbedding, self).get_config()
        config.update({name: getattr(self, name) for name in ["embedding_dim", "num_embeding_layers", "activation"]})
        return config

    def call(self, inputs):
        hidden = inputs
        for layer in self.mlp:
            hidden = layer(hidden)
        return hidden


class CNNEmbedding(tf.keras.layers.Layer):

    def __init__(self, num_layers, layer_size, out_dim, activation, **kwargs):
        super().__init__(**kwargs)
        self.activation, self.num_layers, self.layer_size, self.out_dim = activation, num_layers, layer_size, out_dim

        self.conv_layers = [tf.keras.layers.Conv1D(layer_size, 1) for _ in range(num_layers)]
        self.conv_layers.append(tf.keras.layers.Conv1D(out_dim, 1))
        self.normlizations = [tf.keras.layers.BatchNormalization() for _ in range(num_layers + 1)]
        self.activation_layer = tf.keras.layers.Activation(activation)

    def get_config(self):
        config = super(CNNEmbedding, self).get_config()
        config.update({name: getattr(self, name)
                      for name in ["num_layers", "layer_size", "out_dim", "activation"]})
        return config

    def call(self, inputs):
        # input shape (batch, num_particles, num_particles, feature_dim)
        ones = tf.ones_like(inputs[0, :, :, 0])
        upper_tril_mask = tf.linalg.band_part(ones, 0, -1)
        diag_mask = tf.linalg.band_part(ones, 0, 0)
        upper_tril_mask = tf.cast(upper_tril_mask - diag_mask, tf.bool)
        flattened_upper_triag = tf.boolean_mask(inputs, upper_tril_mask, axis=1)

        hidden = flattened_upper_triag
        for conv, norm in zip(self.conv_layers, self.normlizations):
            hidden = conv(hidden)
            hidden = norm(hidden)
            hidden = self.activation_layer(hidden)

        true_mask = tf.cast(tf.where(upper_tril_mask), tf.int32)
        out = tf.transpose(hidden, [1, 0, 2])
        out = tf.scatter_nd(true_mask, out, shape=[tf.shape(inputs)[1],
                            tf.shape(inputs)[2], tf.shape(inputs)[0], self.out_dim])
        out = out + tf.transpose(out, [1, 0, 2, 3])
        out = tf.transpose(out, [2, 0, 1, 3])
        return out


class DeParTModel(tf.keras.Model):

    def __init__(self,
                 input_shape: Tuple[int],
                 embedding_dim: int,
                 num_embeding_layers: int,
                 layers: int,
                 class_layers: int,
                 expansion: int,
                 heads: int,
                 dropout: float,
                 class_dropout: float,
                 layer_scale_init_value: float,
                 stochastic_depth_drop_rate: float,
                 class_stochastic_depth_drop_rate: float,
                 output_layer: tf.keras.layers.Layer,
                 activation: Callable[[tf.Tensor], tf.Tensor],
                 preprocess: Union[tf.keras.layers.Layer,
                                   Tuple[tf.keras.layers.Layer, tf.keras.layers.Layer], None] = None,
                 interaction_embedding_num_layers: Optional[int] = None,
                 interaction_embedding_layer_size: Optional[int] = None):

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

        hidden = ParticleEmbedding(embedding_dim, num_embeding_layers, activation)(hidden)

        transformed = DeParT(num_selfattn_layers=layers,
                             num_class_layers=class_layers,
                             dim=embedding_dim,
                             expansion=expansion,
                             heads=heads,
                             dropout=dropout,
                             class_dropout=class_dropout,
                             activation=activation,
                             layer_scale_init_value=layer_scale_init_value,
                             stochastic_depth_drop_rate=stochastic_depth_drop_rate,
                             class_stochastic_depth_drop_rate=class_stochastic_depth_drop_rate)(hidden, mask=tf.sequence_mask(row_lengths), interaction=embed_interaction)

        transformed = tf.keras.layers.LayerNormalization()(transformed[:, 0, :])
        output = output_layer(transformed)

        super().__init__(inputs=input, outputs=output)
