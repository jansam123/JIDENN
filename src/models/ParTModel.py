import tensorflow as tf
from typing import Callable, Union, Tuple, Optional


class FFN(tf.keras.layers.Layer):
    def __init__(self, dim, expansion, activation, dropout, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim, self.expansion, self.activation, self.dropout = dim, expansion, activation, dropout
        # Create the required layers -- first a ReLU-activated dense
        # layer with `dim * expansion` units, followed by a dense layer
        # with `dim` units without an activation.
        self.wide_dense = tf.keras.layers.Dense(dim * expansion, activation=activation)
        self.dense = tf.keras.layers.Dense(dim, activation=None)
        self.ln = tf.keras.layers.LayerNormalization()
        self.layer_dropout = tf.keras.layers.Dropout(dropout)

    def get_config(self):
        config = super(FFN, self).get_config()
        config.update({"dim": self.dim, "expansion": self.expansion,
                      "activation": self.activation, "dropout": self.dropout})
        return config

    def call(self, inputs):
        # Execute the FFN Transformer layer.
        output = self.ln(inputs)
        output = self.wide_dense(output)
        output = self.dense(output)
        if self.dropout > 0 and self.dropout is not None:
            output = self.layer_dropout(output)
        return output


class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, dim, heads, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim, self.heads = dim, heads
        # Create weight matrices W_Q, W_K, W_V and W_O using `self.add_weight`,
        # each with shape `[dim, dim]`; for other arguments, keep the default values
        # (which means trainable float32 matrices initialized with `"glorot_uniform"`).
        for name in ["W_Q", "W_K", "W_V", "W_O"]:
            setattr(self, name, self.add_weight(name=name, shape=[dim, dim]))

    def get_config(self):
        config = super(SelfAttention, self).get_config()
        config.update({"dim": self.dim, "heads": self.heads})
        return config

    def call(self, inputs, mask, interaction=None):

        Q, K, V = [
            tf.transpose(
                tf.reshape(
                    inputs @ weights,
                    [tf.shape(inputs)[0], tf.shape(inputs)[1], self.heads, self.dim // self.heads]),
                [0, 2, 1, 3])
            for weights in [self.W_Q, self.W_K, self.W_V]
        ]

        attention_weights = tf.linalg.matmul(Q, K, transpose_b=True) / (Q.shape[-1] ** 0.5)

        if interaction is not None:
            # interaction is of shape [batch_size, N, N, heads]
            interaction = tf.transpose(interaction, [0, 3, 1, 2])
            attention_weights += interaction

        attention = tf.keras.layers.Softmax()(attention_weights, mask=mask)

        hidden = attention @ V
        hidden = tf.transpose(hidden, [0, 2, 1, 3])
        hidden = tf.reshape(hidden, [tf.shape(hidden)[0], tf.shape(hidden)[1], self.dim])
        hidden = hidden @ self.W_O
        return hidden


class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, dim, heads, dropout, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim, self.heads, self.dropout = dim, heads, dropout
        self.mha = tf.keras.layers.MultiHeadAttention(key_dim=dim//heads, num_heads=heads)
        self.ln = tf.keras.layers.LayerNormalization()
        self.layer_dropout = tf.keras.layers.Dropout(dropout)

    def get_config(self):
        config = super(AttentionBlock, self).get_config()
        config.update({"dim": self.dim, "heads": self.heads, "dropout": self.dropout})
        return config

    def call(self, inputs, mask, class_token=None):
        # Execute the Self-Attention Transformer layer.
        if class_token is None:
            output = self.ln(inputs)
            output = self.mha(query=output, value=output, key=output, attention_mask=mask)
        else:
            output = tf.concat([class_token, inputs], axis=1)
            output = self.ln(output)
            output = self.mha(query=class_token, value=output, key=output, attention_mask=mask)
        if self.dropout is not None and self.dropout > 0:
            output = self.layer_dropout(output)
        return output


class InteractionAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, dim, heads, dropout, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim, self.heads, self.dropout = dim, heads, dropout
        self.mha = SelfAttention(dim=dim, heads=heads)
        self.ln = tf.keras.layers.LayerNormalization()
        self.layer_dropout = tf.keras.layers.Dropout(dropout)

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim, "heads": self.heads, "dropout": self.dropout})
        return config

    def call(self, inputs, mask, interaction=None):
        # Execute the Self-Attention Transformer layer.
        output = self.ln(inputs)
        output = self.mha(inputs=output, mask=mask, interaction=interaction)
        # output = self.mha(query=output, value=output, key=output, attention_mask=mask)
        if self.dropout is not None and self.dropout > 0:
            output = self.layer_dropout(output)
        return output


class ParT(tf.keras.layers.Layer):

    def __init__(self, dim, num_particle_layers=8, num_class_layers=2, expansion=4, heads=8, dropout=0.1, activation=tf.nn.gelu, *args, **kwargs):
        # Make sure `dim` is even.
        assert dim % 2 == 0

        super().__init__(*args, **kwargs)
        self.dim, self.expansion, self.heads, self.dropout, self.activation, self.num_particle_layers, self.num_class_layers = dim, expansion, heads, dropout, activation, num_particle_layers, num_class_layers
        # Create the required number of transformer layers, each consisting of
        # - a layer normalization and a self-attention layer followed by a dropout layer,
        # - a layer normalization and a FFN layer followed by a dropout layer.
        self.class_token = tf.Variable(tf.random.truncated_normal((1, 1, dim), stddev=0.02), trainable=True)
        self.particle_layers = [(InteractionAttentionBlock(dim, heads, dropout), FFN(dim, expansion, activation, dropout))
                                for _ in range(num_particle_layers)]
        self.class_layers = [(AttentionBlock(dim, heads, 0.), FFN(dim, expansion, activation, 0.))
                             for _ in range(num_class_layers)]

    def get_config(self):
        config = super(ParT, self).get_config()
        config.update({name: getattr(self, name)
                      for name in ["dim", "expansion", "heads", "dropout", "activation", "num_particle_layers", "num_class_layers"]})
        return config

    def call(self, inputs, mask, interaction=None):
        # Perform the given number of transformer layers, composed of
        # - a self-attention sub-layer, followed by
        # - a FFN sub-layer.
        # In each sub-layer, pass the input through LayerNorm, then compute
        # the corresponding operation, apply dropout, and finally add this result
        # to the original sub-layer input. Note that the given `mask` should be
        # passed to the self-attention operation to ignore the padding words.
        particle_mask = mask[:, tf.newaxis, tf.newaxis, :] & mask[:, tf.newaxis, :, tf.newaxis]
        hidden = inputs
        for attn, ffn in self.particle_layers:
            hidden += attn(hidden, particle_mask, interaction)
            hidden += ffn(hidden)

        query = tf.tile(self.class_token, [tf.shape(inputs)[0], 1, 1])
        class_mask = mask[:, tf.newaxis, :]
        class_mask = tf.concat([tf.ones((tf.shape(inputs)[0], 1, 1), dtype=tf.bool), class_mask], axis=2)
        for attn, ffn in self.class_layers:
            query += attn(hidden, class_mask, query)
            query += ffn(query)
        return query


class ParticleEmbedding(tf.keras.layers.Layer):

    def __init__(self, embedding_dim, num_embeding_layers, activation, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.embedding_dim, self.activation, self.num_embeding_layers = embedding_dim, activation, num_embeding_layers
        self.mlp = tf.keras.Sequential([tf.keras.layers.Dense(self.embedding_dim, activation=self.activation)
                                        for _ in range(self.num_embeding_layers)])

    def get_config(self):
        config = super(ParticleEmbedding, self).get_config()
        config.update({name: getattr(self, name) for name in ["embedding_dim",  "num_embeding_layers", "activation"]})
        return config

    def call(self, inputs):
        hidden = self.mlp(inputs)
        return hidden


class InteractionEmbedding(tf.keras.layers.Layer):

    def __init__(self, num_layers, layer_size, out_dim, activation, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        out = tf.scatter_nd(true_mask, out, shape=[tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[0], self.out_dim])
        out = out + tf.transpose(out, [1, 0, 2, 3])
        out = tf.transpose(out, [2, 0, 1, 3])
        return out


class ParTModel(tf.keras.Model):

    def __init__(self,
                 input_shape: Union[Tuple[int], Tuple[Tuple[int], Tuple[int]]],
                 embedding_dim: int,
                 num_embeding_layers: int,
                 particle_block_layers: int,
                 class_block_layers: int,
                 expansion: int,
                 heads: int,
                 particle_block_dropout: float,
                 output_layer: tf.keras.layers.Layer,
                 activation: Callable[[tf.Tensor], tf.Tensor],
                 interaction: bool = True,
                 interaction_embedding_num_layers: Optional[int] = None,
                 interaction_embedding_layer_size: Optional[int] = None,
                 preprocess: Union[tf.keras.layers.Layer, None, Tuple[tf.keras.layers.Layer, tf.keras.layers.Layer]] = None):

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

        transformed = ParT(dim=embedding_dim,
                           num_particle_layers=particle_block_layers,
                           num_class_layers=class_block_layers,
                           expansion=expansion,
                           heads=heads,
                           dropout=particle_block_dropout,
                           activation=activation)(hidden, tf.sequence_mask(row_lengths), embed_interaction)

        transformed = tf.keras.layers.LayerNormalization()(transformed)
        output = output_layer(transformed[:, 0, :])

        super().__init__(inputs=input, outputs=output)
