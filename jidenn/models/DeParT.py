"""
Implementation of Dynamicaly Ehnanced Particle Transformer (DeParT) model.

This is an extention of ParT model, utilizing several adjustments to the original, namely:

- Talkative Multihead Self-Attention
- LayerScale
- Stochastic Depth
- Gated Feed-Forward Network

All are based on the 'DeiT III: Revenge of the ViT' paper, https://arxiv.org/abs/2204.07118,
except the Gated FFN, https://arxiv.org/abs/2002.05202.

The model also includes the intaraction variables as the ParT model.

![DeParT](images/depart.png)
![DeParT](images/depart_layers.png)

"""
import tensorflow as tf
import keras
from typing import Callable, Union, Tuple, Optional


class RMSNorm(keras.layers.Layer):
    """Root mean square normalization layer
    Root mean square normalization layer normalizes the input tensor along the last dimension,
    and then scales it by a learnable parameter.

    Args:
        dim (int): dimension of the input and output
        eps (float, optional): small value to avoid division by zero. Defaults to 1e-8.
    """

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = keras.Variable(tf.ones((dim,)))

    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps})
        return config

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Forward pass of the root mean square normalization layer

        Args:
            x (tf.Tensor): input tensor of shape `(batch_size, num_particles, dim)`

        Returns:
            tf.Tensor: output tensor of shape `(batch_size, num_particles, dim)`
        """
        return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + self.eps) * self.scale


class FFN(keras.layers.Layer):
    """Feed-forward network
    On top of the Transformer FFN layer, it adds a layer normalization in between the two dense layers.

    On top of ParT FFN layer, it adds a gated linear unit (GLU) activation function.
    This adds additional weights to the layer, so to keep the number of parameters the same,
    the size of the first hidden layer is `dim * expansion * 2 / 3` and the gate 
    hidden layer is the same dimension. 

    Args:
        dim (int): dimension of the input and output
        expansion (int): expansion factor of the hidden layer, i.e. the hidden layer has size `dim * expansion`
        activation (Callable[[tf.Tensor], tf.Tensor]) activation function
        dropout (float, optional): dropout rate. Defaults to None.
    """

    def __init__(self, dim: int, expansion: int, activation: Callable[[tf.Tensor], tf.Tensor], dropout: Optional[float] = None):
        super().__init__()
        self.dim, self.expansion, self.activation, self.dropout = dim, expansion, activation, dropout

        self.wide_dense = keras.layers.Dense(
            int(dim * expansion * 2 / 3), activation=activation, use_bias=False)
        self.gate_dense = keras.layers.Dense(
            int(dim * expansion * 2 / 3), activation=None, use_bias=False)
        self.dense = keras.layers.Dense(
            dim, activation=None, use_bias=False)
        self.ln = keras.layers.LayerNormalization()
        self.layer_dropout = keras.layers.Dropout(dropout)

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim, "expansion": self.expansion,
                      "activation": self.activation, "dropout": self.dropout})
        return config

    def call(self, inputs):
        """Forward pass of the feed-forward network
        Includes a layer normalization layer in between the two dense layers

        Args:
            inputs (tf.Tensor): input tensor of shape `(batch_size, num_particles, dim)`

        Returns:
            tf.Tensor: output tensor of shape `(batch_size, num_particles, dim)`
        """
        output = self.wide_dense(inputs) * self.gate_dense(inputs)
        output = self.ln(output)
        output = self.dense(output)
        output = self.layer_dropout(output)
        return output


class LayerScale(keras.layers.Layer):
    """Layer scale layer
    Layer Scale layer helps to stabilize the training of the model.
    When the model has a large number of layers, the variance of the input to each layer can be very different.
    To stabilize the training, we scale the input to each layer by a learnable scalar parameter,
    which is initialized to a small value.

    Args:
        init_values (float): initial value of the layer scale
        dim (int): dimension of the input and output
    """

    def __init__(self, init_values: float, dim: int, ):
        super().__init__()
        self.gamma = keras.Variable(init_values * tf.ones((dim,)))

    def call(self, x: tf.Tensor, training=False) -> tf.Tensor:
        """Forward pass of the layer scale layer

        Args:
            x (tf.Tensor): input tensor of shape `(batch_size, num_particles, dim)`
        Returns:
            tf.Tensor: output tensor of shape `(batch_size, num_particles, dim)`
        """
        return x * self.gamma


class StochasticDepth(keras.layers.Layer):
    """Stochastic depth layer.

    Stochastic depth is a regularization technique that randomly drops layers instead 
    of individial neurons.

    The probability of dropping should increase with the depth of the layer.
    This must be done manually by the user when creating the layer.

    Args:
        drop_prob (float): probability of dropping the layer
    """

    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = drop_prob

    def call(self, x: tf.Tensor, training=False) -> tf.Tensor:
        """Forward pass of the stochastic depth layer

        Args:
            x (tf.Tensor): input tensor of shape `(batch_size, num_particles, dim)`

        Returns:
            tf.Tensor: output tensor of shape `(batch_size, num_particles, dim)`

        """
        if training:
            keep_prob = 1 - self.drop_prob
            shape = tf.concat([[tf.shape(x)[0]], tf.ones(tf.rank(x) - 1, dtype=tf.int32)], axis=0)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x


class TalkingMultiheadSelfAttention(keras.layers.Layer):
    """Talking Multi-head self-attention layer
    Standalone implementation of the multi-head self-attention layer, which
    includes the interaction variables and the talking heads mechanism.

    Args:
        dim (int): dimension of the input and output
        heads (int): number of heads
        dropout (float, optional): dropout rate. Defaults to None.

    """

    def __init__(self, dim: int, heads: int, dropout: Optional[float] = None, project_interaction: bool = False):
        super().__init__()
        self.dim, self.heads = dim, heads
        self.project_interaction = project_interaction

        self.linear_qkv = keras.layers.Dense(dim * 3)
        self.linear_out = keras.layers.Dense(dim)
        if project_interaction:
            self.interaction_projs = [keras.layers.Dense(
                heads, activation=tf.nn.gelu), keras.layers.Dense(heads, activation=tf.nn.gelu)]

        self.linear_talking_1 = keras.layers.Dense(heads)
        self.linear_talking_2 = keras.layers.Dense(heads)

        self.dropout = keras.layers.Dropout(dropout)
        self.attn_drop = keras.layers.Dropout(dropout)

    def get_config(self):
        config = super(TalkingMultiheadSelfAttention, self).get_config()
        config.update({"dim": self.dim, "heads": self.heads, "dropout": self.dropout,
                      "project_interaction": self.project_interaction})
        return config

    def call(self, inputs: tf.Tensor, mask: tf.Tensor, interaction: Optional[tf.Tensor] = None, training: bool = False) -> tf.Tensor:
        """Forward pass of the talking multi-head self-attention layer

        Args:
            inputs (tf.Tensor): input tensor of shape `(batch_size, num_particles, dim)`
            mask (tf.Tensor): mask tensor of shape `(batch_size, num_particles, num_particles)`
                This mask is used to mask out the attention of padding particles, generated when
                tf.RaggedTensor is converted to tf.Tensor.
            interaction (tf.Tensor, optional): interaction tensor of shape `(batch_size, num_particles, num_particles, heads)`
            training (bool, optional): whether the model is in training mode. Defaults to False.

        Returns:
            tf.Tensor: output tensor of shape `(batch_size, num_particles, dim)`
        """
        B, N, C = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]

        qkv = self.linear_qkv(inputs)  # (B, N, C * 3)
        # (B, N, 3, H, C // H)
        qkv = tf.reshape(qkv, [B, N, 3, self.heads, C // self.heads])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])  # (3, B, H, N, C // H)
        q, k, v = qkv[0], qkv[1], qkv[2]  # 3 x (B, H, N, C // H)

        attention_weights = tf.linalg.matmul(
            q, k, transpose_b=True) / (q.shape[-1] ** 0.5)  # (B, H, N, N)

        attention_weights = self.linear_talking_1(tf.transpose(
            attention_weights, [0, 2, 3, 1]))  # (B, N, N, H)
        attention_weights = tf.transpose(
            attention_weights, [0, 3, 1, 2])  # (B, H, N, N)

        if interaction is not None:
            if self.project_interaction:
                for projection in self.interaction_projs:
                    interaction = projection(interaction)
            interaction = tf.transpose(
                interaction, [0, 3, 1, 2])  # (B, H, N, N)
            attention_weights += interaction

        pure_attention = keras.layers.Softmax()(
            attention_weights, mask=mask)  # (B, H, N, N)
        attention = self.linear_talking_2(tf.transpose(
            pure_attention, [0, 2, 3, 1]))  # (B, N, N, H)
        attention = tf.transpose(attention, [0, 3, 1, 2])  # (B, H, N, N)
        attention = self.attn_drop(attention, training=training)  # (B, H, N, N)

        output = tf.linalg.matmul(attention, v)  # (B, H, N, C // H)
        output = tf.transpose(output, [0, 2, 1, 3])  # (B, N, H, C // H)
        output = tf.reshape(output, [B, N, C])  # (B, N, C)
        output = self.linear_out(output)  # (B, N, C)
        output = self.dropout(output, training=training)
        return output, pure_attention


class TalkingMultiheadClassAttention(keras.layers.Layer):
    """Talking Multi-head class-attention layer
    Standalone implementation of the multi-head class-attention layer, which
    includes the talking heads mechanism.

    Args:
        dim (int): dimension of the input and output
        heads (int): number of heads
        dropout (float, optional): dropout rate, defaults to None

    """

    def __init__(self, dim: int, heads: int, dropout: Optional[float] = None):
        super().__init__()
        self.dim, self.heads = dim, heads

        self.linear_kv = keras.layers.Dense(dim * 2)
        self.linear_q = keras.layers.Dense(dim)
        self.linear_out = keras.layers.Dense(dim)

        self.linear_talking_1 = keras.layers.Dense(heads)
        self.linear_talking_2 = keras.layers.Dense(heads)

        self.dropout = keras.layers.Dropout(dropout)
        self.attn_drop = keras.layers.Dropout(dropout)

    def get_config(self):
        config = super(TalkingMultiheadClassAttention, self).get_config()
        config.update({"dim": self.dim, "heads": self.heads})
        return config

    def call(self, inputs: tf.Tensor, class_token: tf.Tensor, mask: tf.Tensor, training: bool = False, interaction: Optional[tf.Tensor] = None) -> tf.Tensor:
        """Forward pass of the multi-head class-attention layer

        Args:
            inputs (tf.Tensor): input tensor of shape `(batch_size, num_particles, dim)`
            class_token (tf.Tensor): class token tensor of shape `(batch_size, 1, dim)`
            mask (tf.Tensor): mask tensor of shape `(batch_size, 1, num_particles)`
                This mask is used to mask out the attention of padding particles, generated when
                tf.RaggedTensor is converted to tf.Tensor.
            training (bool, optional): whether the model is in training mode. Defaults to False.
        Returns:
            tf.Tensor: output tensor of shape `(batch_size, 1, dim)`
        """

        B, N, C = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]

        kv = self.linear_kv(inputs)  # (B, N, C * 2)
        # (B, N, 2, H, C // H)
        kv = tf.reshape(kv, [B, N, 2, self.heads, C // self.heads])
        kv = tf.transpose(kv, [2, 0, 3, 1, 4])  # (2, B, H, N, C // H)
        k, v = kv[0], kv[1]  # 2 x (B, H, N, C // H)

        q = self.linear_q(class_token)  # (B, 1, C)
        # (B, H, 1, C // H)
        q = tf.reshape(q, [B, self.heads, 1, C // self.heads])

        attention_weights = tf.linalg.matmul(
            q, k, transpose_b=True) / (q.shape[-1] ** 0.5)  # (B, H, 1, N)

        attention_weights = self.linear_talking_1(tf.transpose(
            attention_weights, [0, 2, 3, 1]))  # (B, 1, N, H)
        attention_weights = tf.transpose(
            attention_weights, [0, 3, 1, 2])  # (B, H, 1, N)

        if interaction is not None:
            # interaction shape is (B, N, H)
            interaction = tf.transpose(interaction, [0, 2, 1])  # (B, 1, N)
            interaction = tf.expand_dims(interaction, axis=2)  # (B, 1, 1, N)
            attention_weights += interaction

        mask = tf.expand_dims(mask, axis=1)  # (B, 1, 1, N)
        attention = keras.layers.Softmax()(
            attention_weights, mask=mask)  # (B, H, 1, N)
        attention = self.linear_talking_2(tf.transpose(
            attention, [0, 2, 3, 1]))  # (B, 1, N, H)
        attention = tf.transpose(attention, [0, 3, 1, 2])  # (B, H, 1, N)
        attention = self.attn_drop(attention, training=training)  # (B, H, 1, N)

        output = tf.linalg.matmul(attention, v)  # (B, H, 1, C // H)
        output = tf.transpose(output, [0, 2, 1, 3])  # (B, 1, H, C // H)
        output = tf.reshape(output, [B, 1, C])  # (B, 1, C)
        output = self.linear_out(output)  # (B, 1, C)
        output = self.dropout(output, training=training)
        return output


class MultiheadClassAttention(keras.layers.Layer):
    """Multi-head class attention layer
    This layer is a wrapper around the `keras.layers.MultiHeadAttention` layer, 
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
        self.mha = keras.layers.MultiHeadAttention(
            key_dim=dim // heads, num_heads=heads, dropout=dropout)

    def get_config(self):
        config = super(MultiheadClassAttention, self).get_config()
        config.update({"dim": self.dim, "heads": self.heads,
                      "dropout": self.dropout})
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
        output = self.mha(query=class_token, value=inputs,
                          key=inputs, attention_mask=mask)
        return output


class SelfAttentionBlock(keras.layers.Layer):
    """Self-attention block.
    It contains a talking multi-head self-attention layer and a feed-forward network with residual connections
    and layer normalizations. The self-attention layer includes the interaction variables.
    Additionally, the stochastic dropout and layer scale are applied to the output of the self-attention layer
    and feed-forward network.

    Args:
        dim (int): dimension of the input and output
        heads (int): number of heads
        stoch_drop_prob (float): probability of stochastic dropout
        layer_scale_init_value (float): initial value of layer scale
        activation (Callable[[tf.Tensor], tf.Tensor]) activation function
        expansion (int): expansion factor of the feed-forward network, 
            the dimension of the feed-forward network is `dim * expansion`
        dropout (float, optional): dropout rate, defaults to None

    """

    def __init__(self, dim: int, heads: int, stoch_drop_prob: float, layer_scale_init_value: float, activation: Callable[[tf.Tensor], tf.Tensor], expansion: int,
                 dropout: Optional[float] = None, project_interaction: bool = False):
        super().__init__()
        self.dim, self.heads, self.dropout, self.stoch_drop_prob, self.layer_scale_init_value, self.activation, self.expansion = dim, heads, dropout, stoch_drop_prob, layer_scale_init_value, activation, expansion
        self.project_interaction = project_interaction

        self.pre_mhsa_ln = keras.layers.LayerNormalization()
        self.mhsa = TalkingMultiheadSelfAttention(
            dim, heads, dropout, project_interaction=project_interaction)
        self.post_mhsa_scale = LayerScale(layer_scale_init_value, dim)
        self.post_mhsa_stoch_depth = StochasticDepth(drop_prob=stoch_drop_prob)

        self.pre_ffn_ln = keras.layers.LayerNormalization()
        self.ffn = FFN(dim, expansion, activation, dropout)
        self.post_ffn_scale = LayerScale(layer_scale_init_value, dim)
        self.post_ffn_stoch_depth = StochasticDepth(drop_prob=stoch_drop_prob)

    def get_config(self):
        config = super(SelfAttentionBlock, self).get_config()
        config.update({"dim": self.dim, "heads": self.heads, "dropout": self.dropout, "stoch_drop_prob": self.stoch_drop_prob,
                      "layer_scale_init_value": self.layer_scale_init_value, "activation": self.activation, "expansion": self.expansion, "project_interaction": self.project_interaction})
        return config

    def call(self, inputs, mask, interaction=None):
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
        attended, attention = self.mhsa(attended, mask, interaction)
        attended = self.post_mhsa_scale(attended)
        attended = self.post_mhsa_stoch_depth(attended)
        attended = attended + inputs

        ffned = self.pre_ffn_ln(attended)
        ffned = self.ffn(ffned)
        ffned = self.post_ffn_scale(ffned)
        ffned = self.post_ffn_stoch_depth(ffned)
        output = ffned + attended

        return output, attention


class ClassAttentionBlock(keras.layers.Layer):
    """Class attention block.
    It allows the class token to attend to the input particles, and then feed the attended class token
    to the feed-forward network with residual connections and layer normalizations.

    This extracts the class information from the attended particles more effectively.

    Args:
        dim (int): dimension of the input and output
        heads (int): number of heads
        stoch_drop_prob (float): probability of stochastic dropout
        layer_scale_init_value (float): initial value of layer scale
        activation (Callable[[tf.Tensor], tf.Tensor]) activation function
        expansion (int): expansion factor of the feed-forward network, 
            the dimension of the feed-forward network is `dim * expansion`
        dropout (float, optional): dropout rate, defaults to None
    """

    def __init__(self, dim: int, heads: int, stoch_drop_prob: float, layer_scale_init_value: float, activation: Callable[[tf.Tensor], tf.Tensor], expansion: int,
                 dropout: Optional[float] = None):
        super().__init__()
        self.dim, self.heads, self.dropout, self.stoch_drop_prob, self.layer_scale_init_value, self.activation, self.expansion = dim, heads, dropout, stoch_drop_prob, layer_scale_init_value, activation, expansion

        self.pre_mhca_ln = keras.layers.LayerNormalization()
        self.mhca = TalkingMultiheadClassAttention(dim, heads, dropout)
        self.post_mhca_scale = LayerScale(layer_scale_init_value, dim)
        self.post_mhca_stoch_depth = StochasticDepth(drop_prob=stoch_drop_prob)

        self.pre_ffn_ln = keras.layers.LayerNormalization()
        self.ffn = FFN(dim, expansion, activation, dropout)
        self.post_ffn_scale = LayerScale(layer_scale_init_value, dim)
        self.post_ffn_stoch_depth = StochasticDepth(drop_prob=stoch_drop_prob)

    def get_config(self):
        config = super(ClassAttentionBlock, self).get_config()
        config.update({"dim": self.dim, "heads": self.heads, "dropout": self.dropout, "stoch_drop_prob": self.stoch_drop_prob,
                      "layer_scale_init_value": self.layer_scale_init_value, "activation": self.activation, "expansion": self.expansion})
        return config

    def call(self, inputs: tf.Tensor, class_token: tf.Tensor, mask: tf.Tensor, interaction: Optional[tf.Tensor] = None, training:bool = False) -> tf.Tensor:
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
        interaction = tf.concat([tf.zeros([tf.shape(interaction)[0], 1, tf.shape(
            interaction)[2]]), interaction], axis=1) if interaction is not None else None
        attended = self.pre_mhca_ln(attended)
        attended = self.mhca(class_token=class_token,
                             inputs=attended, mask=mask, interaction=interaction, training=training)
        attended = self.post_mhca_scale(attended)
        attended = self.post_mhca_stoch_depth(attended)
        attended = attended + class_token

        ffned = self.pre_ffn_ln(attended)
        ffned = self.ffn(ffned)
        ffned = self.post_ffn_scale(ffned)
        ffned = self.post_ffn_stoch_depth(ffned)
        output = ffned + attended
        return output


class DeParT(keras.layers.Layer):
    """Pure DeParT layers without the embedding and output layers.

    It also creates the class token, which is used to encode the global information of the input,
    using the ClassAttentionBlock.

    Args:
        dim (int): dimension of the input and output
        self_attn_layers (int): number of self-attention layers
        class_attn_layers (int): number of class-attention layers
        expansion (int): expansion factor of the hidden layer, i.e. the hidden layer has size `dim * expansion`
        heads (int): number of heads
        activation (Callable[[tf.Tensor], tf.Tensor]) activation function
        layer_scale_init_value (float): initial value of layer scale. 
        stochastic_depth_drop_rate (float, optional): drop rate of stochastic depth
        class_stochastic_depth_drop_rate (float, optional): drop rate of stochastic depth of the class token
        dropout (float, optional): dropout rate. Defaults to None.
        class_dropout (float, optional): dropout rate of the class token. Defaults to None.
    """

    def __init__(self,
                 self_attn_layers: int,
                 class_attn_layers: int,
                 dim: int,
                 expansion: int,
                 heads: int,
                 activation: Callable[[tf.Tensor], tf.Tensor],
                 layer_scale_init_value: float,
                 stochastic_depth_drop_rate: Optional[float] = None,
                 class_stochastic_depth_drop_rate: Optional[float] = None,
                 class_dropout: Optional[float] = None,
                 dropout: Optional[float] = None,
                 project_interaction: bool = False):
        # Make sure `dim` is even.
        assert dim % 2 == 0

        super().__init__()
        self.layers, self.dim, self.expansion, self.heads, self.dropout, self.activation, self.class_layers = self_attn_layers, dim, expansion, heads, dropout, activation, class_attn_layers
        self.layer_scale_init_value, self.stochastic_depth_drop_rate, self.class_stochastic_depth_drop_rate = layer_scale_init_value, stochastic_depth_drop_rate, class_stochastic_depth_drop_rate
        self.class_dropout = class_dropout
        self.project_interaction = project_interaction

        self.sa_layers = [SelfAttentionBlock(dim,
                                             heads,
                                             self.stochastic_prob(i, self_attn_layers - 1,
                                                                  stochastic_depth_drop_rate),
                                             layer_scale_init_value,
                                             activation,
                                             expansion,
                                             dropout,
                                             project_interaction) for i in range(self_attn_layers)]

        self.ca_layers = [ClassAttentionBlock(dim,
                                              heads,
                                              self.stochastic_prob(i, class_attn_layers - 1,
                                                                   class_stochastic_depth_drop_rate),
                                              layer_scale_init_value,
                                              activation,
                                              expansion,
                                              class_dropout,) for i in range(class_attn_layers)]

        self.class_token = keras.Variable(tf.random.truncated_normal(
            (1, 1, dim), stddev=0.02), trainable=True, name="class_token")

    def stochastic_prob(self, step, total_steps, drop_rate):
        return drop_rate * step / total_steps

    def get_config(self):
        config = super(DeParT, self).get_config()
        config.update({name: getattr(self, name)
                      for name in ["layers", "dim", "expansion", "heads", "dropout", "activation", "class_layers",
                                   "layer_scale_init_value", "stochastic_depth_drop_rate", "class_stochastic_depth_drop_rate", "class_dropout", "project_interaction"]})
        return config

    def call(self, inputs: tf.Tensor, mask: tf.Tensor, interaction: Optional[tf.Tensor] = None, class_interaction: Optional[tf.Tensor] = None) -> tf.Tensor:
        """Forward pass of the DeParT layers

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
        sa_mask = mask[:, tf.newaxis, tf.newaxis,
                       :] & mask[:, tf.newaxis, :, tf.newaxis]
        hidden = inputs
        attentions = []
        for layer in self.sa_layers:
            hidden, attention = layer(hidden, sa_mask, interaction)
            attentions.append(attention)

        class_token = tf.tile(self.class_token, (tf.shape(inputs)[0], 1, 1))
        class_mask = mask[:, tf.newaxis, :]
        class_mask = tf.concat(
            [tf.ones((tf.shape(inputs)[0], 1, 1), dtype=tf.bool), class_mask], axis=2)
        for layer in self.ca_layers:
            class_token = layer(
                hidden, class_token=class_token, mask=class_mask, interaction=class_interaction)
        return class_token, attentions


class FCEmbedding(keras.layers.Layer):
    """Embedding layer as a series of fully-connected layers.

    Args:
        embed_dim (int): dimension of the embedding
        embed_layers (int): number of fully-connected layers
        activation (Callable[[tf.Tensor], tf.Tensor]) activation function
    """

    def __init__(self, embedding_dim: int, num_embeding_layers: int, activation: Callable[[tf.Tensor], tf.Tensor], ):

        super().__init__()
        self.embedding_dim, self.activation, self.num_embeding_layers = embedding_dim, activation, num_embeding_layers
        self.layers = [keras.layers.Dense(self.embedding_dim, activation=self.activation)
                       for _ in range(self.num_embeding_layers)]

    def get_config(self):
        config = super(FCEmbedding, self).get_config()
        config.update({name: getattr(self, name) for name in [
                      "embedding_dim", "num_embeding_layers", "activation"]})
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


class CNNEmbedding(keras.layers.Layer):
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

        self.conv_layers = [keras.layers.Conv1D(
            layer_size, 1) for _ in range(num_layers)]
        self.conv_layers.append(keras.layers.Conv1D(out_dim, 1))
        self.normlizations = [keras.layers.BatchNormalization()
                              for _ in range(num_layers + 1)]
        self.activation_layer = keras.layers.Activation(activation)

    def get_config(self):
        config = super(CNNEmbedding, self).get_config()
        config.update({name: getattr(self, name)
                      for name in ["num_layers", "layer_size", "out_dim", "activation"]})
        return config

    def call(self, inputs):
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
        flattened_upper_triag = tf.boolean_mask(
            inputs, upper_tril_mask, axis=1)

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

class DeParTModel(keras.Model):
    """DeParT model with embwith embedding and output layers.

    The model already contains the `keras.layers.Input` layer, so it can be used as a standalone model.

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
    Layer normalization is applied to the output of the DeParT layers before the output layer.

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
        layer_scale_init_value (float): initial value of the layer scale parameter
        stochastic_depth_drop_rate (float): drop rate of the stochastic depth regularization
        class_stochastic_depth_drop_rate (float): drop rate of the stochastic depth regularization of the class-attention layers
        output_layer (keras.layers.Layer): output layer
        activation (Callable[[tf.Tensor], tf.Tensor]): activation function
        dropout (Optional[float], optional): dropout rate. Defaults to None.
        class_dropout (Optional[float], optional): dropout rate of the class-attention layers. Defaults to None.
        interaction_embed_layers (Optional[int], optional): number of layers of the interaction embedding layer. Defaults to None.
        interaction_embed_layer_size (Optional[int], optional): size of the layers of the interaction embedding layer. Defaults to None.
        preprocess (Union[keras.layers.Layer, None, Tuple[keras.layers.Layer, keras.layers.Layer]], optional): preprocessing layer. Defaults to None.

    """

    def __init__(self,
                 input_shape: Union[Tuple[None, int], Tuple[Tuple[None, int], Tuple[None, None, int]]],
                 embed_dim: int,
                 embed_layers: int,
                 self_attn_layers: int,
                 class_attn_layers: int,
                 expansion: int,
                 heads: int,
                 layer_scale_init_value: float,
                 stochastic_depth_drop_rate: float,
                 class_stochastic_depth_drop_rate: float,
                 output_layer: keras.layers.Layer,
                 activation: Callable[[tf.Tensor], tf.Tensor],
                 dropout: Optional[float] = None,
                 class_dropout: Optional[float] = None,
                 preprocess: Union[keras.layers.Layer,
                                   Tuple[keras.layers.Layer, keras.layers.Layer], None] = None,
                 interaction_embed_layers: Optional[int] = None,
                 interaction_embed_layer_size: Optional[int] = None,
                 return_attentions: bool = False,
                 **kwargs
                 ):
        self.embed_dim, self.embed_layers, self.self_attn_layers, self.class_attn_layers, self.expansion, self.heads, self.layer_scale_init_value, self.stochastic_depth_drop_rate, self.class_stochastic_depth_drop_rate, self.output_layer, self.activation, self.dropout, self.class_dropout, self.interaction_embed_layers, self.interaction_embed_layer_size, self.return_attentions = embed_dim, embed_layers, self_attn_layers, class_attn_layers, expansion, heads, layer_scale_init_value, stochastic_depth_drop_rate, class_stochastic_depth_drop_rate, output_layer, activation, dropout, class_dropout, interaction_embed_layers, interaction_embed_layer_size, return_attentions
        self.input_size = input_shape
        self.preprocess = preprocess

        if (isinstance(input_shape, tuple) and isinstance(input_shape[0], tuple)) or (isinstance(input_shape, list) and isinstance(input_shape[0], list)):
            if len(input_shape) == 2:
                input = (keras.layers.Input(shape=input_shape[0]),
                         keras.layers.Input(shape=input_shape[1]),
                         keras.layers.Input(shape=(input_shape[0][0],), dtype=tf.bool))
                hidden, interaction_hidden, mask = input
                class_interaction = None
            elif len(input_shape) == 3:
                input = (keras.layers.Input(shape=input_shape[0]),
                         keras.layers.Input(shape=input_shape[1]),
                         keras.layers.Input(shape=input_shape[2]),
                         keras.layers.Input(shape=(input_shape[0][0],), dtype=tf.bool))
                hidden, interaction_hidden, class_interaction, mask = input

            else:
                raise ValueError(
                    "input_shape must be a tuple of two or three tuples.")


            if preprocess is not None:
                if not isinstance(preprocess, tuple) and not isinstance(preprocess, list):
                    raise ValueError(
                        "preprocess must be a tuple of two layers when the input is a tuple of two tensors.")

                preprocess, interaction_preprocess = preprocess
                if interaction_preprocess is not None:
                    interaction_hidden = interaction_preprocess(
                        interaction_hidden)

            if interaction_embed_layers is None or interaction_embed_layer_size is None:
                # if tf.shape(interaction_hidden)[-1] != heads:
                #     raise ValueError(
                #         """The last dimension of the interaction tensor must be equal to the number of heads.
                #         If the interaction tensor is not used, interaction_embed_layers and interaction_embed_layer_size must be specified.""")
                embed_interaction = interaction_hidden
            else:
                embed_interaction = CNNEmbedding(
                    interaction_embed_layers,
                    interaction_embed_layer_size,
                    heads,
                    activation)(interaction_hidden)
        else:
            input = (keras.layers.Input(shape=input_shape),
                     keras.layers.Input(shape=(input_shape[0],), dtype=tf.bool))
            hidden, mask = input
            embed_interaction = None
            class_interaction = None

        if preprocess is not None:
            if isinstance(preprocess, tuple) or isinstance(preprocess, list):
                raise ValueError(
                    "preprocess must be a single layer when the input is a single tensor.")
            hidden = preprocess(hidden)

        hidden = FCEmbedding(embed_dim, embed_layers, activation)(hidden)

        depart_model = DeParT(self_attn_layers=self_attn_layers,
                              class_attn_layers=class_attn_layers,
                              dim=embed_dim,
                              expansion=expansion,
                              heads=heads,
                              dropout=dropout,
                              class_dropout=class_dropout,
                              activation=activation,
                              layer_scale_init_value=layer_scale_init_value,
                              stochastic_depth_drop_rate=stochastic_depth_drop_rate,
                              class_stochastic_depth_drop_rate=class_stochastic_depth_drop_rate,
                              project_interaction=False)#interaction)

        transformed, attentions = depart_model(hidden, mask=mask, interaction=embed_interaction, class_interaction=class_interaction)

        transformed = keras.layers.LayerNormalization()(
            transformed[:, 0, :])
        output = output_layer(transformed)

        if return_attentions:
            super().__init__(inputs=input, outputs=[output, attentions], **kwargs)
        else:
            super().__init__(inputs=input, outputs=output, **kwargs)


    def get_config(self):
        config = super(DeParTModel, self).get_config()
        config.update({name: getattr(self, name)
                      for name in ["embed_dim", "embed_layers", "self_attn_layers", "class_attn_layers",
                                      "expansion", "heads", "layer_scale_init_value", "stochastic_depth_drop_rate", "class_stochastic_depth_drop_rate", "dropout", "class_dropout", "interaction_embed_layers", "interaction_embed_layer_size", "return_attentions"]})
        config['input_shape'] = self.input_size
        config['activation'] = keras.activations.serialize(self.activation)
        config['output_layer'] = keras.saving.serialize_keras_object(self.output_layer)
        if self.preprocess is not None:
            if isinstance(self.preprocess, tuple) or isinstance(self.preprocess, list):
                config['preprocess'] = tuple([keras.saving.serialize_keras_object(layer) for layer in self.preprocess])
            else:
                config['preprocess'] = keras.saving.serialize_keras_object(self.preprocess)
        return config
    
    
    @classmethod
    def from_config(cls, config):
        config['activation'] = keras.activations.deserialize(config['activation'])
        config['output_layer'] = keras.saving.deserialize_keras_object(config['output_layer'])
        if config.get('preprocess') is not None:
            if isinstance(config['preprocess'], tuple) or isinstance(config['preprocess'], list):
                config['preprocess'] = tuple([keras.saving.deserialize_keras_object(layer) for layer in config['preprocess']])
            else:
                config['preprocess'] = keras.saving.deserialize_keras_object(config['preprocess'])
        return cls(**config)