r"""
Implementation of Particle Transformer model based on the paper https://arxiv.org/abs/2202.03772.

This model is a transformer model based on the CaiT transformer model (https://arxiv.org/abs/2103.17239), used in image classification tasks.
On top of the `jidenn.models.Transformer` model, it adds a ClassAttention layers, which allows a more effective feature extraction.
It also includes more layer normalization layers.

The only difference from CaiT, which is a novelty of this model, is the use of **interaction variables**.
These are variables caomputed for each pair of intput particles, which are then used when calculating the attention weights as 
$$ \mathrm{Attention}(Q, K) = \mathrm{softmax} \left( \frac{QK^T}{\sqrt{d_k}} + U \right)$$
where $U$ is the interaction matrix of shape `(batch, num_particles, num_particles, heads)`, where each `head` gets its own interaction matrix.

![ParT](images/part.png)
![ParT](images/part_layers_1.png)
![ParT](images/part_layers_2.png)

"""
import tensorflow as tf
from typing import Callable, Union, Tuple, Optional


class FFN(tf.keras.layers.Layer):
    """Feed-forward network
    On top of the Transformer FFN layer, it adds a layer normalization in between the two dense layers.

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
        self.ln = tf.keras.layers.LayerNormalization()
        self.layer_dropout = tf.keras.layers.Dropout(dropout)

    def get_config(self):
        config = super(FFN, self).get_config()
        config.update({"dim": self.dim, "expansion": self.expansion,
                      "activation": self.activation, "dropout": self.dropout})
        return config

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass of the feed-forward network
        Includes a layer normalization layer in between the two dense layers

        Args:
            inputs (tf.Tensor): input tensor of shape `(batch_size, num_particles, dim)`

        Returns:
            tf.Tensor: output tensor of shape `(batch_size, num_particles, dim)`
        """
        output = self.wide_dense(inputs)
        output = self.ln(output)
        output = self.dense(output)
        output = self.layer_dropout(output)
        return output


class MultiheadSelfAttention(tf.keras.layers.Layer):
    """Multi-head self-attention layer
    Standalone implementation of the multi-head self-attention layer, which
    includes the interaction variables.

    Args:
        dim (int): dimension of the input and output
        heads (int): number of heads

    """

    def __init__(self, dim: int, heads: int, ):
        super().__init__()
        self.dim, self.heads = dim, heads

        self.linear_qkv = tf.keras.layers.Dense(dim * 3)
        self.linear_out = tf.keras.layers.Dense(dim)

    def get_config(self):
        config = super(MultiheadSelfAttention, self).get_config()
        config.update({"dim": self.dim, "heads": self.heads})
        return config

    def call(self, inputs: tf.Tensor, mask: tf.Tensor, interaction: Optional[tf.Tensor] = None) -> tf.Tensor:
        """Forward pass of the multi-head self-attention layer

        Args:
            inputs (tf.Tensor): input tensor of shape `(batch_size, num_particles, dim)`
            mask (tf.Tensor): mask tensor of shape `(batch_size, num_particles, num_particles)`
                This mask is used to mask out the attention of padding particles, generated when
                tf.RaggedTensor is converted to tf.Tensor.
            interaction (tf.Tensor, optional): interaction tensor of shape `(batch_size, num_particles, num_particles, heads)`

        Returns:
            tf.Tensor: output tensor of shape `(batch_size, num_particles, dim)`
        """
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
    """Multi-head class attention layer
    This layer is a wrapper around the `tf.keras.layers.MultiHeadAttention` layer, 
    to fix the key, and value to be the same as the input, and only use the class token
    as the query.

    Args:
        dim (int): dimension of the input and output
        heads (int): number of heads
        dropout (float, optional): dropout rate, defaults to None
    """

    def __init__(self, dim: int, heads: int, dropout: Optional[float] = None):
        super().__init__()
        self.dim, self.heads, self.dropout = dim, heads, dropout
        self.mha = tf.keras.layers.MultiHeadAttention(key_dim=dim // heads, num_heads=heads)
        self.layer_dropout = tf.keras.layers.Dropout(dropout)

    def get_config(self):
        config = super(MultiheadClassAttention, self).get_config()
        config.update({"dim": self.dim, "heads": self.heads, "dropout": self.dropout})
        return config

    def call(self, inputs: tf.Tensor, class_token: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """Forward pass of the multi-head self-attention layer

        Args:
            inputs (tf.Tensor): input tensor of shape `(batch_size, num_particles, dim)`
            class_token (tf.Tensor): class token tensor of shape `(batch_size, 1, dim)`
            mask (tf.Tensor): mask tensor of shape `(batch_size, 1, num_particles)`
                This mask is used to mask out the attention of padding particles, generated when
                tf.RaggedTensor is converted to tf.Tensor.
        Returns:
            tf.Tensor: output tensor of shape `(batch_size, 1, dim)`
        """
        output = self.mha(query=class_token, value=inputs, key=inputs, attention_mask=mask)
        output = self.layer_dropout(output)
        return output


class SelfAttentionBlock(tf.keras.layers.Layer):
    """Self-attention block.
    It contains a multi-head self-attention layer and a feed-forward network with residual connections
    and layer normalizations. The self-attention layer includes the interaction variables.

    Args:
        dim (int): dimension of the input and output
        heads (int): number of heads
        expansion (int): expansion factor of the hidden layer, i.e. the hidden layer has size `dim * expansion`
        activation (Callable[[tf.Tensor], tf.Tensor]) activation function
        dropout (float, optional): dropout rate. Defaults to None.
    """

    def __init__(self, dim: int, heads: int, expansion: int, activation: Callable[[tf.Tensor], tf.Tensor], dropout: Optional[float] = None):
        super().__init__()
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

    def call(self, inputs: tf.Tensor, mask: tf.Tensor, interaction: Optional[tf.Tensor] = None) -> tf.Tensor:
        """Forward pass of the self-attention block

        Args:
            inputs (tf.Tensor): input tensor of shape `(batch_size, num_particles, dim)`
            mask (tf.Tensor, optional): mask tensor of shape `(batch_size, num_particles, num_particles)`. Defaults to None.
                This mask is used to mask out the attention of padding particles, generated when
                tf.RaggedTensor is converted to tf.Tensor.
            interaction (tf.Tensor, optional): interaction tensor of shape `(batch_size, num_particles, num_particles, heads)`. Defaults to None.

        Returns:
            tf.Tensor: output tensor of shape `(batch_size, num_particles, dim)`
        """
        attended = self.pre_mhsa_ln(inputs)
        attended = self.mhsa(inputs=attended, mask=mask, interaction=interaction)
        attended = self.post_mhsa_ln(attended)
        attended = self.mhsa_dropout(attended)
        attended = attended + inputs

        ffned = self.pre_ffn_ln(attended)
        ffned = self.ffn(ffned)
        output = ffned + attended

        return output


class ClassAttentionBlock(tf.keras.layers.Layer):
    """Class attention block.
    It allows the class token to attend to the input particles, and then feed the attended class token
    to the feed-forward network with residual connections and layer normalizations.

    This extracts the class information from the attended particles more effectively.

    Args:
        dim (int): dimension of the input and output
        heads (int): number of heads
        dropout (float, optional): dropout rate. Defaults to None.
        expansion (int): expansion factor of the hidden layer, i.e. the hidden layer has size `dim * expansion`
    """

    def __init__(self, dim: int, heads: int, expansion: int, dropout: Optional[float] = None):
        super().__init__()
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

    def call(self, inputs: tf.Tensor, class_token: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """Forward pass of the class attention block

        Args:
            inputs (tf.Tensor): input tensor of shape `(batch_size, num_particles, dim)`
            class_token (tf.Tensor): class token tensor of shape `(batch_size, 1, dim)`
                It is concatenated with the input tensor along the particle dimension,
                at the front of the input tensor.
            mask (tf.Tensor): mask tensor of shape `(batch_size, 1, num_particles)`
                This mask is used to mask out the attention of padding particles, generated when
                tf.RaggedTensor is converted to tf.Tensor.

        Returns:
            tf.Tensor: output tensor of shape `(batch_size, 1, dim)`, an updated class token
        """
        attended = tf.concat([class_token, inputs], axis=1)
        attended = self.pre_mhca_ln(attended)
        attended = self.mhca(inputs=attended, class_token=class_token, mask=mask)
        attended = self.post_mhca_ln(attended)
        attended = self.mhca_dropout(attended)
        attended = attended + class_token

        ffned = self.pre_ffn_ln(attended)
        ffned = self.ffn(ffned)
        output = ffned + attended
        return output


class ParT(tf.keras.layers.Layer):
    """Pure Particle Transformer (ParT) layers without the embedding and output layers.

    It also creates the class token, which is used to encode the global information of the input,
    using the ClassAttentionBlock.

    Args:
        dim (int): dimension of the input and output
        self_attn_layers (int): number of self-attention layers
        class_attn_layers (int): number of class-attention layers
        expansion (int): expansion factor of the hidden layer, i.e. the hidden layer has size `dim * expansion`
        heads (int): number of heads
        activation (Callable[[tf.Tensor], tf.Tensor]) activation function
        dropout (float, optional): dropout rate. Defaults to None.
    """

    def __init__(self,
                 dim: int,
                 self_attn_layers: int,
                 class_attn_layers: int,
                 expansion: int,
                 heads: int,
                 activation: Callable[[tf.Tensor], tf.Tensor],
                 dropout: Optional[float] = None):
        # Make sure `dim` is even.
        assert dim % 2 == 0

        super().__init__()
        self.dim, self.expansion, self.heads, self.dropout, self.activation, self.num_selfattn_layers, self.num_class_layers = dim, expansion, heads, dropout, activation, self_attn_layers, class_attn_layers

        self.class_token = tf.Variable(tf.random.truncated_normal((1, 1, dim), stddev=0.02), trainable=True)
        self.sa_layers = [SelfAttentionBlock(dim, heads, expansion, activation, dropout)
                          for _ in range(self_attn_layers)]
        self.ca_layers = [ClassAttentionBlock(dim, heads, expansion, dropout) for _ in range(class_attn_layers)]

    def get_config(self):
        config = super(ParT, self).get_config()
        config.update({name: getattr(self, name)
                      for name in ["dim", "expansion", "heads", "dropout", "activation", "num_selfattn_layers", "num_class_layers"]})
        return config

    def call(self, inputs: tf.Tensor, mask: tf.Tensor, interaction: Optional[tf.Tensor] = None) -> tf.Tensor:
        """Forward pass of the ParT layers

        Args:
            inputs (tf.Tensor): input tensor of shape `(batch_size, num_particles, dim)`
            mask (tf.Tensor): mask tensor of shape `(batch_size, num_particles)`.
                From the mask, a mask tensor of shape `(batch_size, num_particles, num_particles)`
                is calculated, which is used to mask out the attention of padding particles, generated when
                `tf.RaggedTensor` is converted to `tf.Tensor`.
            interaction (tf.Tensor, optional): interaction tensor of shape `(batch_size, num_particles, num_particles, heads)`

        Returns:
            tf.Tensor: output tensor of shape `(batch_size, num_particles, dim)`
        """
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


class FCEmbedding(tf.keras.layers.Layer):
    """Embedding layer as a series of fully-connected layers.

    Args:
        embed_dim (int): dimension of the embedding
        embed_layers (int): number of fully-connected layers
        activation (Callable[[tf.Tensor], tf.Tensor]) activation function
    """

    def __init__(self, embedding_dim: int, num_embeding_layers: int, activation: Callable[[tf.Tensor], tf.Tensor], ):

        super().__init__()
        self.embedding_dim, self.activation, self.num_embeding_layers = embedding_dim, activation, num_embeding_layers
        self.layers = [tf.keras.layers.Dense(self.embedding_dim, activation=self.activation)
                       for _ in range(self.num_embeding_layers)]

    def get_config(self):
        config = super(FCEmbedding, self).get_config()
        config.update({name: getattr(self, name) for name in ["embedding_dim", "num_embeding_layers", "activation"]})
        return config

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
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


class CNNEmbedding(tf.keras.layers.Layer):
    """Embedding layer of the interaction variables as a series of point-wise convolutional layers.
    The interaction variiables are compuetd for each pair of particles.
    This creates a redundancy in the input, as the matrix is symetric and the diagonal is always zero.
    To save computation, the upper triangular part of the matrix is used as input, which is 
    flattend and the 1D convolutions are applied to it.

    Args:
        num_layers (int): number of convolutional layers
        layer_size (int): number of channels of the hidden layers
        out_dim (int): number of channels of the last convolutional layer which 
            is manually appended as an extra layer after `num_layers` layers.
        activation (Callable[[tf.Tensor], tf.Tensor]) activation function
    """

    def __init__(self, num_layers: int, layer_size: int, out_dim: int, activation: Callable[[tf.Tensor], tf.Tensor]):
        super().__init__()
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

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass of the interaction embedding layer

        Args:
            inputs (tf.Tensor): input tensor of shape `(batch_size, num_particles, num_particles, num_features)`
                This matrix is assumed to be symetric, with zero diagonal, with only the upper-triag part 
                can be used fopr embedding. 

        Returns:
            tf.Tensor: output tensor of shape `(batch_size, num_particles, num_particles, out_dim)`
        """

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
    """ParT model with embwith embedding and output layers.

    The model already contains the `tf.keras.layers.Input` layer, so it can be used as a standalone model.

    The input tensor can be either a tensor of shape `(batch_size, num_particles, num_features)` or
    a tuple of tensors `(particle_tensor, interaction_tensor)` of shapes 
    `(batch_size, num_particles, num_features)` and `(batch_size, num_particles, num_particles, num_features)`, respectively.

    The model can be used with or without the interaction tensor, depending on the type of the input shape,
    if it is a tuple, the interaction tensor is assumed to be present.

    The input tensor is first passed through the embedding layer, then the ParT layers, and finally the output layer.
    If the interaction tensor is present, it is passed through the interaction embedding layer before the ParT layers.

    If the preprocessing layer is not None, the input tensor is first passed through the preprocessing layer before the embedding layer.
    If the interaction tensor is present, it is passed through the preprocessing layer is an tuple of two layers,
    each of which is applied to the particle and interaction tensors, respectively.

    The output of ParT is a vector of shape `(batch_size, embed_dim)` with extracted class infromation.
    This is then passed through the output layer.
    Layer normalization is applied to the output of the ParT layers before the output layer.

    Args:
        input_shape (Union[Tuple[None, int], Tuple[Tuple[None, int], Tuple[None, None, int]]]): shape of the input tensor.
            If the interaction tensor is present, it is assumed to be a tuple of two shapes,
            each creating a separate input layer.
        embed_dim (int): dimension of the embedding layer
        embed_layers (int): number of layers of the embedding layer
        self_attn_layers (int): number of self-attention layers
        class_attn_layers (int): number of class-attention layers
        expansion (int): expansion factor of the self-attention layers
        heads (int): number of heads of the self-attention layers
        output_layer (tf.keras.layers.Layer): output layer
        activation (Callable[[tf.Tensor], tf.Tensor]): activation function
        dropout (Optional[float], optional): dropout rate. Defaults to None.
        interaction_embed_layers (Optional[int], optional): number of layers of the interaction embedding layer. Defaults to None.
        interaction_embed_layer_size (Optional[int], optional): size of the layers of the interaction embedding layer. Defaults to None.
        preprocess (Union[tf.keras.layers.Layer, None, Tuple[tf.keras.layers.Layer, tf.keras.layers.Layer]], optional): preprocessing layer. Defaults to None.

    """

    def __init__(self,
                 input_shape: Union[Tuple[None, int], Tuple[Tuple[None, int], Tuple[None, None, int]]],
                 embed_dim: int,
                 embed_layers: int,
                 self_attn_layers: int,
                 class_attn_layers: int,
                 expansion: int,
                 heads: int,
                 output_layer: tf.keras.layers.Layer,
                 activation: Callable[[tf.Tensor], tf.Tensor],
                 dropout: Optional[float] = None,
                 interaction_embed_layers: Optional[int] = None,
                 interaction_embed_layer_size: Optional[int] = None,
                 preprocess: Union[tf.keras.layers.Layer, None, Tuple[tf.keras.layers.Layer, tf.keras.layers.Layer]] = None):

        if isinstance(input_shape, tuple) and isinstance(input_shape[0], tuple):
            input = (tf.keras.layers.Input(shape=input_shape[0], ragged=True),
                     tf.keras.layers.Input(shape=input_shape[1], ragged=True))
            row_lengths = input[0].row_lengths()
            hidden = input[0].to_tensor()
            interaction_hidden = input[1].to_tensor()

            if preprocess is not None:
                if not isinstance(preprocess, tuple):
                    raise ValueError(
                        "preprocess must be a tuple of two layers when the input is a tuple of two tensors.")

                preprocess, interaction_preprocess = preprocess
                if interaction_preprocess is not None:
                    interaction_hidden = interaction_preprocess(interaction_hidden)

            if interaction_embed_layers is None or interaction_embed_layer_size is None:
                raise ValueError(
                    """interaction_embed_layers and interaction_embed_layer_size must be specified 
                    when the input is a tuple of two tensors, i.e. the interaction variables are used.""")

            embed_interaction = CNNEmbedding(
                interaction_embed_layers,
                interaction_embed_layer_size,
                heads,
                activation)(interaction_hidden)
        else:
            input = tf.keras.layers.Input(shape=input_shape, ragged=True)
            embed_interaction = None
            row_lengths = input.row_lengths()
            hidden = input.to_tensor()

        if preprocess is not None:
            if isinstance(preprocess, tuple):
                raise ValueError("preprocess must be a single layer when the input is a single tensor.")
            hidden = preprocess(hidden)

        hidden = FCEmbedding(embed_dim, embed_layers, activation)(hidden)

        transformed = ParT(dim=embed_dim,
                           self_attn_layers=self_attn_layers,
                           class_attn_layers=class_attn_layers,
                           expansion=expansion,
                           heads=heads,
                           dropout=dropout,
                           activation=activation)(hidden, tf.sequence_mask(row_lengths), embed_interaction)

        transformed = tf.keras.layers.LayerNormalization()(transformed)
        output = output_layer(transformed[:, 0, :])

        super().__init__(inputs=input, outputs=output)
