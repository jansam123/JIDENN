"""
Implementation of the Transformer model from the paper "Attention is all you need," see https://arxiv.org/abs/1706.03762.

The model is a stack of self-attention blocks, each of which contains a multi-head self-attention layer and a feed-forward network.
The input features are embedded into a vector of size `dim`, which is then passed through the self-attention blocks.

![Transformer](/diagrams/transformer.png)
![Transformer](/diagrams/transformer_layers.png)

"""
import tensorflow as tf
from typing import Callable, Tuple, Optional


class FFN(tf.keras.layers.Layer):
    """Feed-forward network 

    Args:
        dim (int): dimension of the input and output
        expansion (int): expansion factor of the hidden layer, i.e. the hidden layer has size `dim * expansion`
        activation (Callable[[tf.Tensor], tf.Tensor]) activation function
        dropout (float, optional): dropout rate. Defaults to None.
    """

    def __init__(self, dim: int, expansion: int, activation: Callable[[tf.Tensor], tf.Tensor], dropout: Optional[float] = None):
        super().__init__()
        self.dim, self.expansion, self.activation, self.dropout = dim, expansion, activation, dropout

        self.wide_dense = tf.keras.layers.Dense(dim * expansion, activation=activation)
        self.dense = tf.keras.layers.Dense(dim, activation=None)
        self.layer_dropout = tf.keras.layers.Dropout(dropout)

    def get_config(self):
        config = super(FFN, self).get_config()
        config.update({"dim": self.dim, "expansion": self.expansion,
                      "activation": self.activation, "dropout": self.dropout})
        return config

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass of the feed-forward network

        Args:
            inputs (tf.Tensor): input tensor of shape `(batch_size, num_particles, dim)`

        Returns:
            tf.Tensor: output tensor of shape `(batch_size, num_particles, dim)`
        """
        output = self.wide_dense(inputs)
        output = self.dense(output)
        output = self.layer_dropout(output)
        output = self.layer_dropout(output)
        return output


class MultiheadSelfAttention(tf.keras.layers.Layer):
    """Multi-head self-attention layer
    This layer is a wrapper around the `tf.keras.layers.MultiHeadAttention` layer, 
    to fix the key, value, and query to be the same.

    Args:
        dim (int): dimension of the input and output
        heads (int): number of heads

    """

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.dim, self.heads = dim, heads
        self.mha = tf.keras.layers.MultiHeadAttention(key_dim=dim // heads, num_heads=heads)

    def get_config(self):
        config = super(MultiheadSelfAttention, self).get_config()
        config.update({"dim": self.dim, "heads": self.heads})
        return config

    def call(self, inputs: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """Forward pass of the multi-head self-attention layer

        Args:
            inputs (tf.Tensor): input tensor of shape `(batch_size, num_particles, dim)`
            mask (tf.Tensor): mask tensor of shape `(batch_size, num_particles, num_particles)`
                This mask is used to mask out the attention of padding particles, generated when
                tf.RaggedTensor is converted to tf.Tensor.

        Returns:
            tf.Tensor: output tensor of shape `(batch_size, num_particles, dim)`
        """
        output = self.mha(query=inputs, value=inputs, key=inputs, attention_mask=mask)
        return output


class SelfAttentionBlock(tf.keras.layers.Layer):
    """Self-attention block.
    It contains a multi-head self-attention layer and a feed-forward network with residual connections
    and layer normalizations.

    Args:
        dim (int): dimension of the input and output
        heads (int): number of heads
        expansion (int): expansion factor of the hidden layer, i.e. the hidden layer has size `dim * expansion`
        activation (Callable[[tf.Tensor], tf.Tensor]) activation function
        dropout (float, optional): dropout rate. Defaults to None.
    """

    def __init__(self, dim: int, heads: int, expansion: int, activation: Callable[[tf.Tensor], tf.Tensor], dropout: Optional[float] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim, self.heads, self.dropout = dim, heads, dropout
        self.expansion, self.activation = expansion, activation
        self.mhsa_ln = tf.keras.layers.LayerNormalization()
        self.mhsa = MultiheadSelfAttention(dim, heads)
        self.mhsa_dropout = tf.keras.layers.Dropout(dropout)

        self.ffn_ln = tf.keras.layers.LayerNormalization()
        self.ffn = FFN(dim, expansion, activation, dropout)

    def call(self, inputs: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """Forward pass of the self-attention block

        Args:
            inputs (tf.Tensor): input tensor of shape `(batch_size, num_particles, dim)`
            mask (tf.Tensor): mask tensor of shape `(batch_size, num_particles, num_particles)`
                This mask is used to mask out the attention of padding particles, generated when
                tf.RaggedTensor is converted to tf.Tensor.
        Returns:
            tf.Tensor: output tensor of shape `(batch_size, num_particles, dim)`
        """
        attented = self.mhsa_ln(inputs)
        attented = self.mhsa(attented, mask)
        attented = self.mhsa_dropout(attented)
        attented = attented + inputs

        ffned = self.ffn_ln(attented)
        ffned = self.ffn(ffned)
        output = ffned + attented
        return output


class Transformer(tf.keras.layers.Layer):
    """Pure Transformer layers without embedding and output layers.

    It also creates the class token, which is used to encode the global information of the input,
    by concatenating the class token to the input.

    Args:
        layers (int): number of Self-Attention layers
        dim (int): dimension of the input and output
        expansion (int): expansion factor of the hidden layer, i.e. the hidden layer has size `dim * expansion`
        heads (int): number of heads
        activation (Callable[[tf.Tensor], tf.Tensor]) activation function
        dropout (float, optional): dropout rate. Defaults to None.

    """

    def __init__(self, layers: int, dim: int, expansion: int, heads: int, activation: Callable[[tf.Tensor], tf.Tensor], dropout: Optional[float] = None):
        # Make sure `dim` is even.
        assert dim % 2 == 0

        super().__init__()
        self.layers, self.dim, self.expansion, self.heads, self.dropout, self.activation = layers, dim, expansion, heads, dropout, activation
        self.cls_token = tf.Variable(initial_value=tf.random.truncated_normal(
            (1, 1, dim), stddev=0.02), trainable=True)
        self.sa_layers = [SelfAttentionBlock(dim, heads, expansion, activation, dropout) for _ in range(layers)]

    def get_config(self):
        config = super(Transformer, self).get_config()
        config.update({name: getattr(self, name)
                      for name in ["layers", "dim", "expansion", "heads", "dropout", "activation"]})
        return config

    def call(self, inputs: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """Forward pass of the Transformer layers

        Args:
            inputs (tf.Tensor): input tensor of shape `(batch_size, num_particles, dim)`
            mask (tf.Tensor): mask tensor of shape `(batch_size, num_particles)`.
                From the mask, a mask tensor of shape `(batch_size, num_particles, num_particles)`
                is calculated, which is used to mask out the attention of padding particles, generated when
                `tf.RaggedTensor` is converted to `tf.Tensor`. 
        Returns:
            tf.Tensor: output tensor of shape `(batch_size, num_particles, dim)`
        """
        mask = mask[:, tf.newaxis, :] & mask[:, :, tf.newaxis]
        cls_tokens = tf.tile(self.cls_token, [tf.shape(inputs)[0], 1, 1])
        hidden = tf.concat([cls_tokens, inputs], axis=1)
        for sa_block in self.sa_layers:
            hidden = sa_block(hidden, mask)
        return hidden


class FCEmbedding(tf.keras.layers.Layer):
    """Embedding layer as a series of fully-connected layers.

    Args:
        embed_dim (int): dimension of the embedding
        embed_layers (int): number of fully-connected layers
        activation (Callable[[tf.Tensor], tf.Tensor]) activation function
    """

    def __init__(self, embed_dim: int, embed_layers: int, activation: Callable[[tf.Tensor], tf.Tensor]):
        super().__init__()
        self.embedding_dim, self.activation, self.num_embeding_layers = embed_dim, activation, embed_layers
        self.layers = [tf.keras.layers.Dense(self.embedding_dim, activation=self.activation)
                       for _ in range(self.num_embeding_layers)]

    def get_config(self):
        config = super(FCEmbedding, self).get_config()
        config.update({name: getattr(self, name) for name in ["embedding_dim", "num_embeding_layers", "activation"]})
        return config

    def call(self, inputs):
        """Forward pass of the embedding layer

        Args:
            inputs (tf.Tensor): input tensor of shape `(batch_size, num_particles, num_features)`

        Returns:
            tf.Tensor: output tensor of shape `(batch_size, num_particles, embed_dim)`
        """
        hidden = inputs
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class TransformerModel(tf.keras.Model):
    """Transformer model with embedding and output layers.

    The model already contains the `tf.keras.layers.Input` layer, so it can be used as a standalone model.

    The input tensor is first passed through the embedding layer, then the Transformer layers, and finally the output layer.
    If the preprocessing layer is not None, the input tensor is first passed through the preprocessing layer before the embedding layer.
    The input to the output layer is the extracted class token.  

    Args:
        input_shape (Tuple[int]): shape of the input
        embed_dim (int): dimension of the embedding
        embed_layers (int): number of fully-connected layers in the embedding
        self_attn_layers (int): number of Self-Attention layers
        expansion (int): expansion factor of the hidden layer, i.e. the hidden layer has size `dim * expansion`
        heads (int): number of heads
        dropout (float, optional): dropout rate. Defaults to None.
        output_layer (tf.keras.layers.Layer): output layer
        activation (Callable[[tf.Tensor], tf.Tensor]) activation function used in all the layers
        preprocess (tf.keras.layers.Layer, optional): preprocessing layer. Defaults to None.
    """

    def __init__(self,
                 input_shape: Tuple[None, int],
                 embed_dim: int,
                 embed_layers: int,
                 self_attn_layers: int,
                 expansion: int,
                 heads: int,
                 dropout: float,
                 output_layer: tf.keras.layers.Layer,
                 activation: Callable[[tf.Tensor], tf.Tensor],
                 preprocess: Optional[tf.keras.layers.Layer] = None):

        input = tf.keras.layers.Input(shape=input_shape, ragged=True)

        row_lengths = input.row_lengths()
        hidden = input.to_tensor()

        if preprocess is not None:
            hidden = preprocess(hidden)

        hidden = FCEmbedding(embed_dim, embed_layers, activation)(hidden)
        row_lengths += 1

        transformed = Transformer(self_attn_layers, embed_dim, expansion,
                                  heads, activation, dropout)(hidden, mask=tf.sequence_mask(row_lengths))

        transformed = tf.keras.layers.LayerNormalization()(transformed[:, 0, :])
        output = output_layer(transformed)

        super().__init__(inputs=input, outputs=output)
