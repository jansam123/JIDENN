import tensorflow as tf
from typing import Callable, Union


class FFN(tf.keras.layers.Layer):
    def __init__(self, dim, expansion=4, dropout=0.1, activation=tf.nn.gelu, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim, self.expansion, self.activation, self.dropout = dim, expansion, activation, dropout
        # Create the required layers -- first a ReLU-activated dense
        # layer with `dim * expansion` units, followed by a dense layer
        # with `dim` units without an activation.
        self.wide_dense = tf.keras.layers.Dense(dim * expansion, activation=activation)
        self.dense = tf.keras.layers.Dense(dim, activation=None)
        self.preln = tf.keras.layers.LayerNormalization()
        self.postln = tf.keras.layers.LayerNormalization()
        self.dropout_layer = tf.keras.layers.Dropout(dropout)

    def get_config(self):
        config = super(FFN, self).get_config()
        config.update({"dim": self.dim, "expansion": self.expansion, "activation": self.activation})
        return config

    def call(self, inputs):
        # Execute the FFN Transformer layer.
        output = self.preln(inputs)
        output = self.wide_dense(output)
        output = self.dropout_layer(output)
        output = self.postln(output)
        output = self.dense(output)
        return output


class P_MHA(tf.keras.layers.Layer):
    def __init__(self, dim, heads=8, dropout=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim, self.heads, self.dropout = dim, heads, dropout
        self.preln = tf.keras.layers.LayerNormalization()
        self.mha = tf.keras.layers.MultiHeadAttention(key_dim=dim, num_heads=heads, dropout=dropout)
        self.postln = tf.keras.layers.LayerNormalization()

    def get_config(self):
        config = super(P_MHA, self).get_config()
        config.update({"dim": self.dim, "heads": self.heads, "dropout": self.dropout})
        return config

    def call(self, inputs, mask):
        # Execute the Self-Attention Transformer layer.
        inputs = self.preln(inputs)
        output = self.mha(query=inputs, value=inputs, key=inputs, attention_mask=mask)
        output = self.postln(output)
        return output


class MHA(tf.keras.layers.Layer):
    def __init__(self, dim, heads=8, dropout=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim, self.heads, self.dropout = dim, heads, dropout
        self.preln = tf.keras.layers.LayerNormalization()
        self.mha = tf.keras.layers.MultiHeadAttention(key_dim=dim, num_heads=heads, dropout=dropout)
        self.postln = tf.keras.layers.LayerNormalization()

    def get_config(self):
        config = super(MHA, self).get_config()
        config.update({"dim": self.dim, "heads": self.heads, "dropout": self.dropout})
        return config

    def call(self, query, inputs, mask):
        # Execute the Self-Attention Transformer layer.
        inputs = self.preln(inputs)
        output = self.mha(query=query, value=inputs, key=inputs, attention_mask=mask)
        output = self.postln(output)
        return output


class ParticleAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, dim, heads=8, expansion=4, dropout=0.1, activation=tf.nn.gelu, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim, self.heads, self.expansion, self.activation, self.dropout = dim, heads, expansion, activation, dropout
        self.p_mha = P_MHA(dim, heads, dropout)
        self.ffn = FFN(dim, expansion, dropout, activation)
        self.layer_dropout = tf.keras.layers.Dropout(dropout)

    def get_config(self):
        config = super(ParticleAttentionBlock, self).get_config()
        config.update({name: getattr(self, name)
                      for name in ["layers", "dim", "expansion", "heads", "dropout", "activation"]})
        return config

    def call(self, inputs, mask):
        attned = self.p_mha(inputs, mask)
        attned = inputs + attned
        attned = self.layer_dropout(attned)
        output = self.ffn(attned)
        output = output + attned
        output = self.layer_dropout(output)
        return output


class ClassAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, dim, heads=8, expansion=4, activation=tf.nn.gelu, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim, self.heads, self.expansion, self.activation = dim, heads, expansion, activation
        self.mha = MHA(dim, heads, 0.)
        self.ffn = FFN(dim, expansion, 0., activation)

    def get_config(self):
        config = super(ClassAttentionBlock, self).get_config()
        config.update({name: getattr(self, name)
                      for name in ["layers", "dim", "expansion", "heads", "dropout", "activation"]})
        return config

    def call(self, class_token, inputs, mask):
        inputs = tf.concat([class_token, inputs], axis=1)
        attned = self.mha(class_token, inputs, mask)
        attned = class_token + attned
        output = self.ffn(attned)
        output = output + attned
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
        self.class_token = tf.Variable(tf.random.normal((1, 1, dim)), trainable=True)
        self.particle_layers = [ParticleAttentionBlock(
            dim, heads, expansion, dropout, activation) for _ in range(num_particle_layers)]
        self.class_layers = [ClassAttentionBlock(dim, heads, expansion, activation) for _ in range(num_class_layers)]

    def get_config(self):
        config = super(ParT, self).get_config()
        config.update({name: getattr(self, name)
                      for name in ["dim", "expansion", "heads", "dropout", "activation", "num_particle_layers", "num_class_layers"]})
        return config

    def call(self, inputs, mask):
        # Perform the given number of transformer layers, composed of
        # - a self-attention sub-layer, followed by
        # - a FFN sub-layer.
        # In each sub-layer, pass the input through LayerNorm, then compute
        # the corresponding operation, apply dropout, and finally add this result
        # to the original sub-layer input. Note that the given `mask` should be
        # passed to the self-attention operation to ignore the padding words.
        particle_mask = mask[:, tf.newaxis, :] & mask[:, :, tf.newaxis]
        hidden = inputs
        for particle_layer in self.particle_layers:
            hidden = particle_layer(hidden, particle_mask)
        query = tf.tile(self.class_token, [tf.shape(inputs)[0], 1, 1])
        class_mask = mask[:, tf.newaxis, :]
        class_mask = tf.concat([tf.ones((tf.shape(inputs)[0], 1, 1), dtype=tf.bool), class_mask], axis=2)
        for class_layer in self.class_layers:
            query = class_layer(query, hidden, class_mask)
        return query


class ParticleEmbedding(tf.keras.layers.Layer):

    def __init__(self, embedding_dim, num_embeding_layers, activation, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.embedding_dim, self.activation, self.num_embeding_layers = embedding_dim, activation, num_embeding_layers
        self.mlp = tf.keras.Sequential([tf.keras.layers.LayerNormalization(),
                                        tf.keras.layers.Dense(self.embedding_dim),
                                        tf.keras.layers.Activation(self.activation),
                                        tf.keras.layers.LayerNormalization(),
                                        tf.keras.layers.Dense(4*self.embedding_dim),
                                        tf.keras.layers.Activation(self.activation),
                                        tf.keras.layers.LayerNormalization(),
                                        tf.keras.layers.Dense(self.embedding_dim),
                                        tf.keras.layers.Activation(self.activation),
                                        ])

    def get_config(self):
        config = super(ParticleEmbedding, self).get_config()
        config.update({name: getattr(self, name) for name in ["embedding_dim",  "num_embeding_layers", "activation"]})
        return config

    def call(self, inputs):
        hidden = self.mlp(inputs)
        return hidden


class ParTModel(tf.keras.Model):

    def __init__(self,
                 input_shape: tuple[int],
                 embedding_dim: int,
                 num_embeding_layers: int,
                 particle_block_layers: int,
                 class_block_layers: int,
                 transformer_expansion: int,
                 transformer_heads: int,
                 particle_block_dropout: float,
                 output_layer: tf.keras.layers.Layer,
                 activation: Callable[[tf.Tensor], tf.Tensor],
                 preprocess: Union[tf.keras.layers.Layer, None] = None):

        input = tf.keras.layers.Input(shape=input_shape, ragged=True)

        row_lengths = input.row_lengths()
        hidden = input.to_tensor()

        if preprocess is not None:
            hidden = preprocess(hidden)

        hidden = ParticleEmbedding(embedding_dim, num_embeding_layers, activation)(hidden)
        transformed = ParT(embedding_dim, particle_block_layers, class_block_layers, transformer_expansion,
                           transformer_heads, particle_block_dropout, activation)(hidden, tf.sequence_mask(row_lengths))
        transformed = tf.squeeze(transformed, axis=1)
        transformed = tf.keras.layers.LayerNormalization()(transformed)
        output = output_layer(transformed)

        super().__init__(inputs=input, outputs=output)
