import tensorflow as tf
from typing import Callable, Union, Tuple


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
        output = self.layer_dropout(output)
        return output


class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, dim, heads, dropout, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim, self.heads, self.dropout = dim, heads, dropout
        self.mha = tf.keras.layers.MultiHeadAttention(key_dim=dim // heads, num_heads=heads)
        self.ln = tf.keras.layers.LayerNormalization()
        self.layer_dropout = tf.keras.layers.Dropout(dropout)

    def get_config(self):
        config = super(SelfAttention, self).get_config()
        config.update({"dim": self.dim, "heads": self.heads, "dropout": self.dropout})
        return config

    def call(self, inputs, mask):
        # Execute the Self-Attention Transformer layer.
        output = self.ln(inputs)
        output = self.mha(query=output, value=output, key=output, attention_mask=mask)
        output = self.layer_dropout(output)
        return output


class Transformer(tf.keras.layers.Layer):

    def __init__(self, layers, dim, expansion, heads, dropout, activation, *args, **kwargs):
        # Make sure `dim` is even.
        assert dim % 2 == 0

        super().__init__(*args, **kwargs)
        self.layers, self.dim, self.expansion, self.heads, self.dropout, self.activation = layers, dim, expansion, heads, dropout, activation
        # Create the required number of transformer layers, each consisting of
        # - a layer normalization and a self-attention layer followed by a dropout layer,
        # - a layer normalization and a FFN layer followed by a dropout layer.
        self.layers = []
        for _ in range(layers):
            self.layers.append([
                SelfAttention(dim=dim, heads=heads, dropout=dropout),
                FFN(dim, expansion, activation, dropout),
            ])

    def get_config(self):
        config = super(Transformer, self).get_config()
        config.update({name: getattr(self, name)
                      for name in ["layers", "dim", "expansion", "heads", "dropout", "activation"]})
        return config

    def call(self, inputs, mask):
        # Perform the given number of transformer layers, composed of
        # - a self-attention sub-layer, followed by
        # - a FFN sub-layer.
        # In each sub-layer, pass the input through LayerNorm, then compute
        # the corresponding operation, apply dropout, and finally add this result
        # to the original sub-layer input. Note that the given `mask` should be
        # passed to the self-attention operation to ignore the padding words.
        mask = mask[:, tf.newaxis, :] & mask[:, :, tf.newaxis]
        hidden = inputs
        for attn, ffn in self.layers:
            hidden += attn(hidden, mask)
            hidden += ffn(hidden)
        return hidden


class ParticleEmbedding(tf.keras.layers.Layer):

    def __init__(self, embedding_dim, num_embeding_layers, activation, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.embedding_dim, self.activation, self.num_embeding_layers = embedding_dim, activation, num_embeding_layers
        self.cls_token = tf.Variable(initial_value=tf.random.truncated_normal(
            (1, 1, self.embedding_dim), stddev=0.02), trainable=True)
        self.mlp = tf.keras.Sequential([tf.keras.layers.Dense(self.embedding_dim, activation=self.activation)
                                        for _ in range(self.num_embeding_layers)])

    def get_config(self):
        config = super(ParticleEmbedding, self).get_config()
        config.update({name: getattr(self, name) for name in ["embedding_dim", "num_embeding_layers", "activation"]})
        return config

    def call(self, inputs):
        hidden = self.mlp(inputs)
        cls_tokens = tf.tile(self.cls_token, [tf.shape(inputs)[0], 1, 1])
        hidden = tf.concat([cls_tokens, hidden], axis=1)
        return hidden


class TransformerModel(tf.keras.Model):

    def __init__(self,
                 input_shape: Tuple[int],
                 embedding_dim: int,
                 num_embeding_layers: int,
                 layers: int,
                 expansion: int,
                 heads: int,
                 dropout: float,
                 output_layer: tf.keras.layers.Layer,
                 activation: Callable[[tf.Tensor], tf.Tensor],
                 preprocess: Union[tf.keras.layers.Layer, None] = None):

        input = tf.keras.layers.Input(shape=input_shape, ragged=True)

        row_lengths = input.row_lengths()
        hidden = input.to_tensor()

        if preprocess is not None:
            hidden = preprocess(hidden)

        hidden = ParticleEmbedding(embedding_dim, num_embeding_layers, activation)(hidden)
        row_lengths += 1

        transformed = Transformer(layers, embedding_dim, expansion,
                                  heads, dropout, activation)(hidden, mask=tf.sequence_mask(row_lengths))

        transformed = tf.keras.layers.LayerNormalization()(transformed[:, 0, :])
        output = output_layer(transformed)

        super().__init__(inputs=input, outputs=output)
