import tensorflow as tf
from src.config.ArgumentParser import ArgumentParser
from typing import Optional, Callable


class TransformerModel(tf.keras.Model):
    class FFN(tf.keras.layers.Layer):
        def __init__(self, dim, expansion, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.dim, self.expansion = dim, expansion
            # TODO: Create the required layers -- first a ReLU-activated dense
            # layer with `dim * expansion` units, followed by a dense layer
            # with `dim` units without an activation.
            self.wide_dense = tf.keras.layers.Dense(dim * expansion, activation=tf.nn.relu)
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
            self.W_Q = self.add_weight(shape=(dim, dim))
            self.W_K = self.add_weight(shape=(dim, dim))
            self.W_V = self.add_weight(shape=(dim, dim))
            self.W_O = self.add_weight(shape=(dim, dim))

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
            Z = tf.linalg.matmul(Q, K, transpose_b=True) / tf.math.sqrt(float(self.dim // self.heads))

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
        def __init__(self, layers, dim, expansion, heads, dropout, *args, **kwargs):
            # Make sure `dim` is even.
            assert dim % 2 == 0

            super().__init__(*args, **kwargs)
            self.layers, self.dim, self.expansion, self.heads, self.dropout = layers, dim, expansion, heads, dropout
            # TODO: Create the required number of transformer layers, each consisting of
            # - a layer normalization and a self-attention layer followed by a dropout layer,
            # - a layer normalization and a FFN layer followed by a dropout layer.
            self.FFN = [TransformerModel.FFN(dim, expansion) for _ in range(layers)]
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
                 args: ArgumentParser, 
                 input_layer: tf.keras.layers.Layer,
                 output_layer: tf.keras.layers.Layer, 
                 metrics: list[tf.keras.metrics.Metric],
                 loss: tf.keras.losses.Loss,
                 preprocess: tf.keras.layers.Layer | None = None):
        # Implement a transformer encoder network. The input `words` is
        # a RaggedTensor of strings, each batch example being a list of words.
        input = input_layer
        if preprocess is not None:
            hidden = preprocess(input)
        else:
            hidden = input
    
        hidden = tf.keras.layers.Embedding(input_dim=args.input_size, output_dim=args.embed_dim)(hidden)

        # TODO: Call the Transformer layer:
        # - create a `Model.Transformer` layer, using suitable options from `args`
        #   (using `args.we_dim` for the `dim` argument),
        # - when calling the layer, convert the ragged tensor with the input words embedding
        #   to a dense one, and also pass the following argument as a mask:
        #     `mask=tf.sequence_mask(ragged_tensor_with_input_words_embeddings.row_lengths())`
        # - finally, convert the result back to a ragged tensor.
        transformed = TransformerModel.Transformer(args.transformer_layers, args.embed_dim, args.transformer_expansion, args.transformer_heads, args.transformer_dropout)(hidden)

        # TODO(tagger_we): Add a softmax classification layer into as many classes as there are unique
        # tags in the `word_mapping` of `train.tags`. Note that the Dense layer can process
        # a `RaggedTensor` without any problem.
        transformed = tf.keras.layers.Flatten()(transformed)
        transformed = tf.keras.layers.Dense(args.last_fc_size, activation=tf.nn.relu)(transformed)
        output = output_layer(transformed)
    

        
        class LinearWarmup(tf.optimizers.schedules.LearningRateSchedule):
            def __init__(self, warmup_steps, following_schedule):
                self._warmup_steps = warmup_steps
                self._warmup = tf.optimizers.schedules.PolynomialDecay(0., warmup_steps, following_schedule(0))
                self._following = following_schedule

            def __call__(self, step):
                return tf.cond(step < self._warmup_steps,
                            lambda: self._warmup(step),
                            lambda: self._following(step - self._warmup_steps))
                
        super().__init__(inputs=input, outputs=output)
        l_r = LinearWarmup(args.warmup_steps, tf.optimizers.schedules.CosineDecay(args.learning_rate, args.decay_steps))
        l_r = 0.001
        self.compile(optimizer=tf.optimizers.Adam(learning_rate=l_r),  # type: ignore
                                loss=loss,
                                weighted_metrics=metrics,)

        