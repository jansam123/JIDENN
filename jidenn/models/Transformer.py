"""
Implementation of the Transformer model from the paper "Attention is all you need," see https://arxiv.org/abs/1706.03762.

The model is a stack of self-attention blocks, each of which contains a multi-head self-attention layer and a feed-forward network.
The input features are embedded into a vector of size `dim`, which is then passed through the self-attention blocks.

![Transformer](images/transformer.png)
![Transformer](images/transformer_layers.png)

"""
import tensorflow as tf
import keras
from typing import Callable, Tuple, Optional


class FFN(keras.layers.Layer):
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

        self.wide_dense = keras.layers.Dense(dim * expansion, activation=activation)
        self.dense = keras.layers.Dense(dim, activation=None)
        self.layer_dropout = keras.layers.Dropout(dropout)

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
        return output


class MultiheadSelfAttention(keras.layers.Layer):
    """Multi-head self-attention layer
    This layer is a wrapper around the `keras.layers.MultiHeadAttention` layer, 
    to fix the key, value, and query to be the same.

    Args:
        dim (int): dimension of the input and output
        heads (int): number of heads

    """

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.dim, self.heads = dim, heads
        self.mha = keras.layers.MultiHeadAttention(key_dim=dim // heads, num_heads=heads)

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


class SelfAttentionBlock(keras.layers.Layer):
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
        self.mhsa_ln = keras.layers.LayerNormalization()
        self.mhsa = MultiheadSelfAttention(dim, heads)
        self.mhsa_dropout = keras.layers.Dropout(dropout)

        self.ffn_ln = keras.layers.LayerNormalization()
        self.ffn = FFN(dim, expansion, activation, dropout)
    
    def get_config(self):
        config = super(SelfAttentionBlock, self).get_config()
        config.update({"dim": self.dim, "heads": self.heads, "expansion": self.expansion,
                      "activation": self.activation, "dropout": self.dropout})
        return config

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
        attended = self.mhsa_ln(inputs)
        attended = self.mhsa(attended, mask)
        attended = self.mhsa_dropout(attended)
        attended = attended + inputs

        ffned = self.ffn_ln(attended)
        ffned = self.ffn(ffned)
        output = ffned + attended
        return output


class Transformer(keras.layers.Layer):
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
        self.class_token = keras.Variable(tf.random.truncated_normal(
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
        mask = tf.concat([tf.ones_like(mask[:, :1]), mask], axis=1)
        mask = mask[:, tf.newaxis, :] & mask[:, :, tf.newaxis]
        class_tokens = tf.tile(self.class_token, [tf.shape(inputs)[0], 1, 1])
        hidden = tf.concat([class_tokens, inputs], axis=1)
        for sa_block in self.sa_layers:
            hidden = sa_block(hidden, mask)
        return hidden


class FCEmbedding(keras.layers.Layer):
    """Embedding layer as a series of fully-connected layers.

    Args:
        embed_dim (int): dimension of the embedding
        embed_layers (int): number of fully-connected layers
        activation (Callable[[tf.Tensor], tf.Tensor]) activation function
    """

    def __init__(self, embed_dim: int, embed_layers: int, activation: Callable[[tf.Tensor], tf.Tensor]):
        super().__init__()
        self.embedding_dim, self.activation, self.num_embeding_layers = embed_dim, activation, embed_layers
        self.layers = [keras.layers.Dense(self.embedding_dim, activation=self.activation)
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


class TransformerModel(keras.Model):
    """Transformer model with embedding and output layers.

    The model already contains the `keras.layers.Input` layer, so it can be used as a standalone model.

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
        output_layer (keras.layers.Layer): output layer
        activation (Callable[[tf.Tensor], tf.Tensor]) activation function used in all the layers
        preprocess (keras.layers.Layer, optional): preprocessing layer. Defaults to None.
    """

    def __init__(self,
                 input_shape: Tuple[None, int],
                 embed_dim: int,
                 embed_layers: int,
                 self_attn_layers: int,
                 expansion: int,
                 heads: int,
                 dropout: float,
                 output_layer: keras.layers.Layer,
                 activation: Callable[[tf.Tensor], tf.Tensor],
                 preprocess: Optional[keras.layers.Layer] = None,
                 **kwargs
                 ):
        
        self.input_size, self.embed_dim, self.embed_layers, self.self_attn_layers, self.expansion, self.heads, self.dropout, self.output_layer, self.activation, self.preprocess = input_shape, embed_dim, embed_layers, self_attn_layers, expansion, heads, dropout, output_layer, activation, preprocess

        input = (keras.layers.Input(shape=input_shape),
                keras.layers.Input(shape=(input_shape[0],), dtype=tf.bool))

        hidden = input[0]
        mask = input[1]

        if preprocess is not None:
            hidden = preprocess(hidden)

        hidden = FCEmbedding(embed_dim, embed_layers, activation)(hidden)
        
        

        transformed = Transformer(self_attn_layers, embed_dim, expansion,
                                  heads, activation, dropout)(hidden, mask=mask)

        transformed = keras.layers.LayerNormalization()(transformed[:, 0, :])
        output = output_layer(transformed)

        super().__init__(inputs=input, outputs=output, **kwargs)
        
    def get_config(self):
        config = super(TransformerModel, self).get_config()
        config.update({name: getattr(self, name) for name in ["embed_dim", "embed_layers", "self_attn_layers", "expansion", "heads", "dropout"]})
        config["output_layer"] = keras.layers.serialize(self.output_layer)
        config["activation"] = keras.activations.serialize(self.activation)
        config["preprocess"] = keras.layers.serialize(self.preprocess)
        config["input_shape"] = self.input_size
        return config
    
    @classmethod
    def from_config(cls, config):
        config["output_layer"] = keras.layers.deserialize(config["output_layer"])
        config["activation"] = keras.activations.deserialize(config["activation"])
        config["preprocess"] = keras.layers.deserialize(config["preprocess"])
        return cls(**config)

        
