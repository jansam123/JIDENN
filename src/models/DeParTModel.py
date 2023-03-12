
import tensorflow as tf
from typing import Callable, Union, Tuple, Optional


class FFN(tf.keras.layers.Layer):
    def __init__(self, dim, expansion, activation, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dim, self.expansion, self.activation, self.dropout = dim, expansion, activation, dropout
        # Create the required layers -- first a ReLU-activated dense
        # layer with `dim * expansion` units, followed by a dense layer
        # with `dim` units without an activation.
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
        # Execute the FFN Transformer layer.
        output = self.wide_dense(inputs) * self.gate_dense(inputs)
        output = self.dense(output)
        output = self.layer_dropout(output)
        return output


class LayerScale(tf.keras.layers.Layer):
    """LayerScale as introduced in CaiT: https://arxiv.org/abs/2103.17239.

    Args:
        init_values (float): value to initialize the diagonal matrix of LayerScale.
        projection_dim (int): projection dimension used in LayerScale.
    Reference:
        https://keras.io/examples/vision/deit
    """

    def __init__(self, init_values: float, projection_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.gamma = tf.Variable(init_values * tf.ones((projection_dim,)))

    def call(self, x, training=False):
        return x * self.gamma


class StochasticDepth(tf.keras.layers.Layer):
    """Stochastic Depth layer (https://arxiv.org/abs/1603.09382).

    Reference:
        https://github.com/rwightman/pytorch-image-models
        https://keras.io/examples/vision/deit
    """

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


class TalkingHeadAttention(tf.keras.layers.Layer):
    """Talking-head attention as proposed in CaiT: https://arxiv.org/abs/2003.02436.

    Args:
        projection_dim (int): projection dimension for the query, key, and value
            of attention.
        num_heads (int): number of attention heads.
        dropout_rate (float): dropout rate to be used for dropout in the attention
            scores as well as the final projected outputs.
    Reference:
        https://keras.io/examples/vision/deit
    """

    def __init__(self, dim: int, heads: int, **kwargs):
        super().__init__(**kwargs)

        self.num_heads = heads
        head_dim = dim // self.num_heads

        self.scale = head_dim**-0.5

        self.qkv = tf.keras.layers.Dense(dim * 3)
        # self.attn_drop = tf.keras.layers.Dropout(dropout)

        self.proj = tf.keras.layers.Dense(dim)

        self.proj_l = tf.keras.layers.Dense(self.num_heads)
        self.proj_w = tf.keras.layers.Dense(self.num_heads)

        # self.proj_drop = tf.keras.layers.Dropout(dropout)

    def call(self, x, mask, interaction=None, training=False):
        B, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]

        # Project the inputs all at once.
        qkv = self.qkv(x)

        # Reshape the projected output so that they're segregated in terms of
        # query, key, and value projections.
        qkv = tf.reshape(qkv, (B, N, 3, self.num_heads, C // self.num_heads))

        # Transpose so that the `num_heads` becomes the leading dimensions.
        # Helps to better segregate the representation sub-spaces.
        qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])  # 3, B, num_heads, N, C // num_heads
        scale = tf.cast(self.scale, dtype=qkv.dtype)
        q, k, v = qkv[0] * scale, qkv[1], qkv[2]  # B, num_heads, N, C // num_heads

        # Permute the key to match the shape of the query.
        k = tf.transpose(k, perm=[0, 1, 3, 2])  # B, num_heads, C // num_heads, N
        # Obtain the raw attention scores.
        attn = tf.matmul(q, k)  # B, num_heads, N, N

        # Linear projection of the similarities between the query and key projections.
        attn = self.proj_l(tf.transpose(attn, perm=[0, 2, 3, 1]))

        # Normalize the attention scores.
        attn = tf.transpose(attn, perm=[0, 3, 1, 2])

        if interaction is not None:
            interaction = tf.transpose(interaction, perm=[0, 3, 1, 2])
            attn += interaction

        attn = tf.keras.layers.Softmax()(attn, mask=mask)

        # Linear projection on the softmaxed scores.
        attn = self.proj_w(tf.transpose(attn, perm=[0, 2, 3, 1]))
        attn = tf.transpose(attn, perm=[0, 3, 1, 2])
        # attn = self.attn_drop(attn, training)

        # Final set of projections as done in the vanilla attention mechanism.
        x = tf.matmul(attn, v)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, (B, N, C))

        x = self.proj(x)
        # x = self.proj_drop(x, training)

        return x


class ClassAttention(tf.keras.layers.Layer):
    def __init__(self, dim, heads, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dim, self.heads, self.dropout = dim, heads, dropout
        self.mha = tf.keras.layers.MultiHeadAttention(key_dim=dim // heads, num_heads=heads, dropout=dropout)

    def get_config(self):
        config = super(ClassAttention, self).get_config()
        config.update({"dim": self.dim, "heads": self.heads, "dropout": self.dropout})
        return config

    def call(self, query, inputs, mask):
        # Execute the Self-Attention Transformer layer.
        output = self.mha(query=query, value=inputs, key=inputs, attention_mask=mask)
        return output


class ClassAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, dim, heads, dropout, stoch_drop_prob, layer_scale_init_value, activation, expansion, **kwargs):
        super().__init__(**kwargs)
        self.dim, self.heads, self.dropout, self.stoch_drop_prob, self.layer_scale_init_value, self.activation, self.expansion = dim, heads, dropout, stoch_drop_prob, layer_scale_init_value, activation, expansion
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.attn = ClassAttention(dim, heads, dropout)
        self.stoch_depth1 = StochasticDepth(drop_prob=stoch_drop_prob)
        self.layer_scale1 = LayerScale(layer_scale_init_value, dim)
        self.stoch_depth2 = StochasticDepth(drop_prob=stoch_drop_prob)
        self.layer_scale2 = LayerScale(layer_scale_init_value, dim)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mlp = FFN(dim, expansion, activation, dropout)

    def get_config(self):
        config = super(ClassAttentionBlock, self).get_config()
        config.update({"dim": self.dim, "heads": self.heads, "dropout": self.dropout, "stoch_drop_prob": self.stoch_drop_prob,
                      "layer_scale_init_value": self.layer_scale_init_value, "activation": self.activation, "expansion": self.expansion})
        return config

    def call(self, inputs, cls_token, mask):
        inputs = tf.concat([cls_token, inputs], axis=1)
        mask = tf.concat([tf.ones((tf.shape(inputs)[0], 1, 1), dtype=tf.bool), mask], axis=2)
        # Execute the Self-Attention Transformer layer.
        output1 = self.norm1(inputs)
        output1 = self.attn(query=cls_token, inputs=output1, mask=mask)
        output1 = self.layer_scale1(output1)
        output1 = self.stoch_depth1(output1)
        output1 = output1 + cls_token
        # Execute the Feed-Forward Transformer layer.
        output2 = self.norm2(output1)
        output2 = self.mlp(output2)
        output2 = self.layer_scale2(output2)
        output2 = self.stoch_depth2(output2)
        output2 = output2 + output1
        return output2


class SelfAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, dim, heads, dropout, stoch_drop_prob, layer_scale_init_value, activation, expansion, **kwargs):
        super().__init__(**kwargs)
        self.dim, self.heads, self.dropout, self.stoch_drop_prob, self.layer_scale_init_value, self.activation, self.expansion = dim, heads, dropout, stoch_drop_prob, layer_scale_init_value, activation, expansion
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.attn = TalkingHeadAttention(dim, heads)
        # self.attn = tf.keras.layers.MultiHeadAttention(key_dim=dim, num_heads=heads)
        self.stoch_depth1 = StochasticDepth(drop_prob=stoch_drop_prob)
        self.layer_scale1 = LayerScale(layer_scale_init_value, dim)
        self.stoch_depth2 = StochasticDepth(drop_prob=stoch_drop_prob)
        self.layer_scale2 = LayerScale(layer_scale_init_value, dim)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = FFN(dim, expansion, activation, dropout)

    def get_config(self):
        config = super(SelfAttentionBlock, self).get_config()
        config.update({"dim": self.dim, "heads": self.heads, "dropout": self.dropout, "stoch_drop_prob": self.stoch_drop_prob,
                      "layer_scale_init_value": self.layer_scale_init_value, "activation": self.activation, "expansion": self.expansion})
        return config

    def call(self, inputs, mask, interaction=None):
        # Execute the Self-Attention Transformer layer.
        output1 = self.norm1(inputs)
        output1 = self.attn(output1, mask, interaction)
        # output1 = self.attn(output1, output1, output1, mask)
        output1 = self.layer_scale1(output1)
        output1 = self.stoch_depth1(output1)
        output1 = output1 + inputs
        # Execute the Feed-Forward Transformer layer.
        output2 = self.norm2(output1)
        output2 = self.ffn(output2)
        output2 = self.layer_scale2(output2)
        output2 = self.stoch_depth2(output2)
        output2 = output2 + output1
        return output2


class DeParT(tf.keras.layers.Layer):
    def __init__(self,
                 layers: int,
                 class_layers: int,
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
        self.layers, self.dim, self.expansion, self.heads, self.dropout, self.activation, self.class_layers = layers, dim, expansion, heads, dropout, activation, class_layers
        self.layer_scale_init_value, self.stochastic_depth_drop_rate, self.class_stochastic_depth_drop_rate = layer_scale_init_value, stochastic_depth_drop_rate, class_stochastic_depth_drop_rate
        self.class_dropout = class_dropout
        # Create the required number of transformer layers, each consisting of
        # - a layer normalization and a self-attention layer followed by a dropout layer,
        # - a layer normalization and a FFN layer followed by a dropout layer.
        self.layers = [SelfAttentionBlock(dim, heads, dropout, self.stochastic_prob(i, layers - 1, stochastic_depth_drop_rate),
                                          layer_scale_init_value, activation, expansion) for i in range(layers)]

        self.class_layers = [ClassAttentionBlock(dim, heads, class_dropout, self.stochastic_prob(i, class_layers - 1, class_stochastic_depth_drop_rate),
                                                 layer_scale_init_value, activation, expansion) for i in range(class_layers)]
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
        # Create a token for the class.
        # Execute the transformer layers.
        self_mask = mask[:, tf.newaxis, tf.newaxis, :] & mask[:, tf.newaxis, :, tf.newaxis]
        for layer in self.layers:
            inputs = layer(inputs, self_mask, interaction)

        cls_token = tf.tile(self.cls_token, (tf.shape(inputs)[0], 1, 1))
        class_mask = mask[:, tf.newaxis, :]
        for layer in self.class_layers:
            cls_token = layer(inputs, cls_token=cls_token, mask=class_mask)
        return cls_token


class ParticleEmbedding(tf.keras.layers.Layer):

    def __init__(self, embedding_dim, num_embeding_layers, activation, **kwargs):

        super().__init__(**kwargs)
        self.embedding_dim, self.activation, self.num_embeding_layers = embedding_dim, activation, num_embeding_layers
        self.mlp = tf.keras.Sequential([tf.keras.layers.Dense(self.embedding_dim, activation=self.activation)
                                        for _ in range(self.num_embeding_layers)])

    def get_config(self):
        config = super(ParticleEmbedding, self).get_config()
        config.update({name: getattr(self, name) for name in ["embedding_dim", "num_embeding_layers", "activation"]})
        return config

    def call(self, inputs):
        # hidden = self.concat([self.mlp(inputs), self.cnns(inputs)])
        hidden = self.mlp(inputs)
        return hidden


class InteractionEmbedding(tf.keras.layers.Layer):

    def __init__(self, num_layers, layer_size, out_dim, activation, **kwargs):
        super().__init__(**kwargs)
        self.activation = activation
        self.layer_size = layer_size
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.conv_layers = [tf.keras.layers.Conv1D(layer_size, 1) for _ in range(num_layers)]
        self.conv_layers.append(tf.keras.layers.Conv1D(out_dim, 1))
        self.normlizations = [tf.keras.layers.BatchNormalization() for _ in range(num_layers + 1)]
        self.activation_layer = tf.keras.layers.Activation(activation)

    def get_config(self):
        config = super(InteractionEmbedding, self).get_config()
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
                 interaction: bool = True,
                 interaction_embedding_num_layers: Optional[int] = None,
                 interaction_embedding_layer_size: Optional[int] = None):

        if interaction:
            input = (tf.keras.layers.Input(shape=input_shape[0], ragged=True),
                     tf.keras.layers.Input(shape=input_shape[1], ragged=True))
            row_lengths = input[0].row_lengths()
            hidden = input[0].to_tensor()
            interaction_hidden = input[1].to_tensor()

            if preprocess is not None:
                preprocess, interaction_preprocess = preprocess
                if interaction_preprocess is not None:
                    interaction_hidden = interaction_preprocess(interaction_hidden)

            embed_interaction = InteractionEmbedding(
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

        transformed = DeParT(layers=layers,
                             class_layers=class_layers,
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
