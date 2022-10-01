import tensorflow as tf

from typing import Callable


class TransformerModel(tf.keras.Model):
    class FFN(tf.keras.layers.Layer):
        def __init__(self, dim, expansion, activation, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.dim, self.expansion = dim, expansion
            # TODO: Create the required layers -- first a ReLU-activated dense
            # layer with `dim * expansion` units, followed by a dense layer
            # with `dim` units without an activation.
            self.wide_dense = tf.keras.layers.Dense(dim * expansion, activation=activation)
            self.dense = tf.keras.layers.Dense(dim, activation=None)

        def get_config(self):
            return {"dim": self.dim, "expansion": self.expansion}

        def call(self, inputs):
            # TODO: Execute the FFN Transformer layer.
            hidden = self.wide_dense(inputs)
            output = self.dense(hidden)
            return output

    class SelfAttention(tf.keras.layers.Layer):
        def __init__(self, dim, heads, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.dim, self.heads = dim, heads
            # TODO: Create weight matrices W_Q, W_K, W_V and W_O using `self.add_weight`,
            # each with shape `[dim, dim]`; for other arguments, keep the default values
            # (which mean trainable float32 matrices initialized with `"glorot_uniform"`).
            self.W_Q = self.add_weight(shape=(dim, dim), name="W_Q")
            self.W_K = self.add_weight(shape=(dim, dim), name="W_K")
            self.W_V = self.add_weight(shape=(dim, dim), name="W_V")
            self.W_O = self.add_weight(shape=(dim, dim), name="W_O")

        def get_config(self):
            return {"dim": self.dim, "heads": self.heads}

        def call(self, inputs):
            # TODO: Execute the self-attention layer.
            #
            # Start by computing Q, K and V. In all cases:
            # - first multiply `inputs` by the corresponding weight matrix W_Q/W_K/W_V,
            # - reshape via `tf.reshape` to [batch_size, max_sentence_len, heads, dim // heads],
            # - transpose via `tf.transpose` to [batch_size, heads, max_sentence_len, dim // heads].
            QKV = []

            for matrix in [self.W_Q, self.W_K, self.W_V]:
                shaped = tf.reshape(inputs @ matrix, (tf.shape(inputs)
                                    [0], tf.shape(inputs)[1], self.heads, self.dim // self.heads))
                QKV.append(tf.transpose(shaped, [0, 2, 1, 3]))

            Q, K, V = QKV

            # TODO: Continue by computing the self-attention weights as Q @ K^T,
            # normalizing by the square root of `dim // heads`.
            Z = tf.linalg.matmul(Q, K, transpose_b=True) / (Q.shape[-1] ** 0.5)

            # TODO: Apply the softmax, but including a suitable mask, which ignores all padding words.
            # The original `mask` is a bool matrix of shape [batch_size, max_sentence_len]
            # indicating which words are valid (True) or padding (False).
            # - You can perform the masking manually, by setting the attention weights
            #   of padding words to -1e9.
            # - Alternatively, you can use the fact that tf.keras.layers.Softmax accepts a named
            #   boolean argument `mask` indicating the valid (True) or padding (False) elements.
            Z = tf.keras.layers.Softmax()(Z)

            # TODO: Finally,
            # - take a weighted combination of values V according to the computed attention
            #   (using a suitable matrix multiplication),
            # - transpose the result to [batch_size, max_sentence_len, heads, dim // heads],
            # - reshape to [batch_size, max_sentence_len, dim],
            # - multiply the result by the W_O matrix.
            Z = tf.linalg.matmul(Z, V)
            Z = tf.transpose(Z, [0, 2, 1, 3])
            Z = tf.reshape(Z, (tf.shape(inputs)[0], tf.shape(inputs)[1], self.dim))
            output = Z @ self.W_O
            return output

    class Transformer(tf.keras.layers.Layer):
        def __init__(self, layers, dim, expansion, heads, dropout, activation, *args, **kwargs):
            # Make sure `dim` is even.
            assert dim % 2 == 0

            super().__init__(*args, **kwargs)
            self.layers, self.dim, self.expansion, self.heads, self.dropout = layers, dim, expansion, heads, dropout
            # TODO: Create the required number of transformer layers, each consisting of
            # - a layer normalization and a self-attention layer followed by a dropout layer,
            # - a layer normalization and a FFN layer followed by a dropout layer.
            self.FFN = [TransformerModel.FFN(dim, expansion, activation) for _ in range(layers)]
            self.self_attention = [TransformerModel.SelfAttention(dim, heads) for _ in range(layers)]
            self.dropout = tf.keras.layers.Dropout(dropout)
            self.layer_norms_fnn = [tf.keras.layers.LayerNormalization() for _ in range(layers)]
            self.layer_norms_attn = [tf.keras.layers.LayerNormalization() for _ in range(layers)]
            self.add = tf.keras.layers.Add()

        def get_config(self):
            return {name: getattr(self, name) for name in ["layers", "dim", "expansion", "heads", "dropout"]}

        def call(self, inputs):
            # Perform the given number of transformer layers, composed of
            # - a self-attention sub-layer, followed by
            # - a FFN sub-layer.
            # In each sub-layer, pass the input through LayerNorm, then compute
            # the corresponding operation, apply dropout, and finally add this result
            # to the original sub-layer input. 
            encoded = inputs
            for ffn, self_att, layer_norm_fnn, layer_norm_attn in zip(self.FFN, self.self_attention, self.layer_norms_fnn, self.layer_norms_attn):
                attended = layer_norm_attn(encoded)
                attended = self_att(attended)
                attended = self.add([encoded, self.dropout(attended)])

                encoded = layer_norm_fnn(attended)
                encoded = ffn(encoded)
                encoded = self.add([attended, self.dropout(encoded)])

            return encoded
        
    def __init__(self, 
                 input_size: int,
                 embedding_dim: int,
                 transformer_layers: int,
                 transformer_expansion: int, 
                 transformer_heads: int, 
                 transformer_dropout: float,
                 last_hidden_layer: int,
                 output_layer: tf.keras.layers.Layer, 
                 metrics: list[tf.keras.metrics.Metric],
                 loss: tf.keras.losses.Loss,
                 optimizer:tf.keras.optimizers.Optimizer,
                 activation:Callable[[tf.Tensor], tf.Tensor],
                 preprocess: tf.keras.layers.Layer | None = None):
        # Implement a transformer encoder network. The input `words` is
        # a RaggedTensor of strings, each batch example being a list of words.
        input = tf.keras.layers.Input(shape=(input_size))
        if preprocess is not None:
            hidden = preprocess(input)
        else:
            hidden = input
    
        hidden = tf.keras.layers.Embedding(input_dim=input_size, output_dim=embedding_dim)(hidden)

        # TODO: Call the Transformer layer:
        # - create a `Model.Transformer` layer, using suitable options from `args`
        #   (using `args.we_dim` for the `dim` argument),
        # - when calling the layer, convert the ragged tensor with the input words embedding
        #   to a dense one, and also pass the following argument as a mask:
        #     `mask=tf.sequence_mask(ragged_tensor_with_input_words_embeddings.row_lengths())`
        # - finally, convert the result back to a ragged tensor.
        transformed = TransformerModel.Transformer(transformer_layers, embedding_dim, transformer_expansion, transformer_heads, transformer_dropout, activation)(hidden)

        # TODO(tagger_we): Add a softmax classification layer into as many classes as there are unique
        # tags in the `word_mapping` of `train.tags`. Note that the Dense layer can process
        # a `RaggedTensor` without any problem.
        transformed = tf.keras.layers.Flatten()(transformed)
        transformed = tf.keras.layers.Dense(last_hidden_layer, activation=activation)(transformed)
        output = output_layer(transformed)
                
        super().__init__(inputs=input, outputs=output)
        
        self.compile(optimizer=optimizer,  # type: ignore
                                loss=loss,
                                weighted_metrics=metrics,)

        