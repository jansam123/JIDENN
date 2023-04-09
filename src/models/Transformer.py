import tensorflow as tf
from typing import Callable, Union, Tuple, Optional


class FFN(tf.keras.layers.Layer):
    def __init__(self, dim: int, expansion: int, activation: Callable, dropout: Optional[float] = None, *args, **kwargs):
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
        output = self.ln(inputs)
        output = self.wide_dense(output)
        output = self.dense(output)
        if self.dropout > 0 and self.dropout is not None:
            output = self.layer_dropout(output)
        return output


class MultiheadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, dim: int, heads: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim, self.heads = dim, heads
        self.mha = tf.keras.layers.MultiHeadAttention(key_dim=dim // heads, num_heads=heads)

    def get_config(self):
        config = super(MultiheadSelfAttention, self).get_config()
        config.update({"dim": self.dim, "heads": self.heads})
        return config

    def call(self, inputs, mask):
        output = self.mha(query=inputs, value=inputs, key=inputs, attention_mask=mask)
        return output


class SelfAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, dim: int, heads: int, expansion: int, activation: Callable, dropout: Optional[float] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim, self.heads, self.dropout = dim, heads, dropout
        self.expansion, self.activation = expansion, activation
        self.mhsa_ln = tf.keras.layers.LayerNormalization()
        self.mhsa = MultiheadSelfAttention(dim, heads)
        self.mhsa_dropout = tf.keras.layers.Dropout(dropout)

        self.ffn_ln = tf.keras.layers.LayerNormalization()
        self.ffn = FFN(dim, expansion, activation, dropout)

    def call(self, inputs, mask):
        attented = self.mhsa_ln(inputs)
        attented = self.mhsa(attented, mask)
        attented = self.mhsa_dropout(attented)
        attented = attented + inputs

        ffned = self.ffn_ln(attented)
        ffned = self.ffn(ffned)
        output = ffned + attented
        return output


class Transformer(tf.keras.layers.Layer):

    def __init__(self, layers: int, dim: int, expansion: int, heads: int, activation: Callable, dropout: Optional[float] = None, *args, **kwargs):
        # Make sure `dim` is even.
        assert dim % 2 == 0

        super().__init__(*args, **kwargs)
        self.layers, self.dim, self.expansion, self.heads, self.dropout, self.activation = layers, dim, expansion, heads, dropout, activation
        self.cls_token = tf.Variable(initial_value=tf.random.truncated_normal(
            (1, 1, self.embedding_dim), stddev=0.02), trainable=True)
        self.sa_layers = [SelfAttentionBlock(dim, heads, expansion, activation, dropout) for _ in range(layers)]

    def get_config(self):
        config = super(Transformer, self).get_config()
        config.update({name: getattr(self, name)
                      for name in ["layers", "dim", "expansion", "heads", "dropout", "activation"]})
        return config

    def call(self, inputs, mask):
        mask = mask[:, tf.newaxis, :] & mask[:, :, tf.newaxis]
        cls_tokens = tf.tile(self.cls_token, [tf.shape(inputs)[0], 1, 1])
        hidden = tf.concat([cls_tokens, inputs], axis=1)
        for sa_block in self.sa_layers:
            hidden = sa_block(hidden)
        return hidden


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

    def call(self, inputs):
        hidden = inputs
        for layer in self.mlp:
            hidden = layer(hidden)
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

        hidden = Embedding(embedding_dim, num_embeding_layers, activation)(hidden)
        row_lengths += 1

        transformed = Transformer(layers, embedding_dim, expansion,
                                  heads, dropout, activation)(hidden, mask=tf.sequence_mask(row_lengths))

        transformed = tf.keras.layers.LayerNormalization()(transformed[:, 0, :])
        output = output_layer(transformed)

        super().__init__(inputs=input, outputs=output)
