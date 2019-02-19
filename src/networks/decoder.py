import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer


class DecoderLayer(tf.layers.Layer):
    def __init__(self, params, mode:tf.estimator.ModeKeys, dtype=tf.float32, name="encoder"):
        super(DecoderLayer, self).__init__(True, name, dtype)
        self.params = params
        self.mode = mode

    def build(self, input_shape):
        # To calculate attention, we calculate
        #   v^T tanh(W_h h_i + W_s s_t + b_attn)
        # where h_i is an encoder state, and s_t a decoder state.
        # attn_vec_size is the length of the vectors v, b_attn, (W_h h_i) and (W_s s_t).
        # We set it to be equal to the size of the encoder states.
        self.attention_vec_size = 2 * self.params.hidden_dim

        self.cell = tf.nn.rnn_cell.LSTMCell(self.params.hidden_dim, state_is_tuple=True, initializer=xavier_initializer())

        # Get the weight matrix W_h and apply it to each encoder state to get (W_h h_i), the encoder features
        self.W_h = tf.get_variable("W_h", [1, 1, self.attention_vec_size, self.attention_vec_size], dtype=tf.float32, initializer=xavier_initializer())

        # Get the weight vectors v and w_c (w_c is for coverage)
        self.v = tf.get_variable("v", [self.attention_vec_size])
        if self.params.coverage:
            with tf.variable_scope("coverage"):
                self.w_c = tf.get_variable("w_c", [1, 1, 1, self.attention_vec_size], dtype=tf.float32, initializer=xavier_initializer())

        self.linear_proj = tf.layers.Dense(self.attention_vec_size, use_bias=True, kernel_initializer=xavier_initializer())
        self.linear_proj2 = tf.layers.Dense(self.params.embedding_size, use_bias=True, kernel_initializer=xavier_initializer())
        self.linear_proj3 = tf.layers.Dense(1, use_bias=True, kernel_initializer=xavier_initializer())
        self.linear_proj4 = tf.layers.Dense(self.params.hidden_dim, use_bias=True, kernel_initializer=xavier_initializer())
        self.built = True

    def call(self, inputs, **kwargs):
        encoder_states = inputs["enc_states"]
        enc_mask = inputs["enc_mask"]
        batch_size = self.params.batch_size

        # now is shape (batch_size, attn_len, 1, attention_vec_size)
        encoder_states = tf.expand_dims(encoder_states, axis=2)
        # shape (batch_size,attn_length,1,attention_vec_size)
        encoder_features = tf.nn.conv2d(encoder_states, self.W_h, [1, 1, 1, 1], "SAME")

        def attention(decoder_state, coverage=None):
            with tf.variable_scope("Attention"):
                # Pass the decoder state through a linear layer (this is W_s s_t + b_attn in the paper)
                # shape (batch_size, attention_vec_size)
                decoder_features = self.linear_proj(tf.concat(decoder_state, axis=1))
                # reshape to (batch_size, 1, 1, attention_vec_size)
                decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1)

                def masked_attention(e):
                    adder = (1.0 - tf.cast(enc_mask, tf.float32)) * -10000.0
                    e += adder
                    # take softmax. shape (batch_size, attn_length)
                    attn_dist = tf.nn.softmax(e)
                    return attn_dist

                if self.params.coverage and coverage is not None: # non-first step of coverage
                    # Multiply coverage vector by w_c to get coverage_features.
                    coverage_features = tf.nn.conv2d(coverage, self.w_c, [1, 1, 1, 1], "SAME") # c has shape (batch_size, attn_length, 1, attention_vec_size)

                    # Calculate v^T tanh(W_h h_i + W_s s_t + w_c c_i^t + b_attn)
                    e = tf.reduce_sum(self.v * tf.tanh(encoder_features + decoder_features + coverage_features), [2, 3])  # shape (batch_size,attn_length)

                    # Calculate attention distribution
                    attn_dist = masked_attention(e)

                    # Update coverage vector
                    coverage += tf.reshape(attn_dist, [batch_size, -1, 1, 1])
                else:
                    # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
                    # encoder_features + decoder_features, [batch_size, sequence_length, 1, attention_vec_size]
                    # v * tanh(), [batch_size, sequence_length, 1, attention_vec_size]
                    # e [batch_size, sequence_length]
                    e = tf.reduce_sum(self.v * tf.tanh(encoder_features + decoder_features), [2, 3]) # calculate e

                    # Calculate attention distribution
                    # attn_dist [batch_size, sequence_length]
                    attn_dist = masked_attention(e)

                    if self.params.coverage: # first step of training
                        coverage = tf.expand_dims(tf.expand_dims(attn_dist,2),2) # initialize coverage

                # Calculate the context vector from attn_dist and encoder_states
                # shape (batch_size, attention_vec_size).
                context_vector = tf.reduce_sum(
                    tf.reshape(attn_dist, [batch_size, -1, 1, 1]) * encoder_states,
                    [1, 2])
                context_vector = tf.reshape(context_vector, [-1, self.attention_vec_size])

            return context_vector, attn_dist, coverage

        outputs = []
        attn_dists = []
        p_gens = []
        state = inputs["dec_in_state"]
        coverage = inputs.get("prev_coverage", None)
        context_vector = tf.zeros([batch_size, self.attention_vec_size])
        if self.mode == tf.estimator.ModeKeys.PREDICT:
            # Re-calculate the context vector from the previous step so that we can pass it through a linear layer with this step's input to get a modified version of the input
            context_vector, _, coverage = attention(inputs["dec_in_state"], coverage)  # in decode mode, this is what updates the coverage vector

        for i, inp in enumerate(inputs[self.params.dec_input_name]):
            tf.logging.info("Adding attention_decoder timestep %i of %i", i, len(inputs[self.params.dec_input_name]))

            # Merge input and previous attentions into one vector x of the same size as inp
            # shape of inp: [batch_size, embedding_size]
            # shape of context_vector: [batch_size, 2*self.params.hidden_dim]
            x = self.linear_proj2(tf.concat([inp, context_vector], axis=1))

            # Run the decoder RNN cell. cell_output = decoder state
            cell_output, state = self.cell(x, state)

            # Run the attention mechanism.
            if i == 0 and self.mode == tf.estimator.ModeKeys.PREDICT:  # always true in decode mode
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    context_vector, attn_dist, _ = attention(state, coverage)  # don't allow coverage to update
            else:
                context_vector, attn_dist, coverage = attention(state, coverage)
            attn_dists.append(attn_dist)

            # Calculate p_gen
            if self.params.pointer_gen:
                with tf.variable_scope('calculate_pgen'):
                    # Tensor shape (batch_size, 1)
                    p_gen = self.linear_proj3(tf.concat([context_vector, state.c, state.h, x], axis=1))
                    p_gen = tf.sigmoid(p_gen)
                    p_gens.append(p_gen)

            # Concatenate the cell_output (= decoder state) and the context vector, and pass them through a linear layer
            # This is V[s_t, h*_t] + b in the paper
            with tf.variable_scope("AttnOutputProjection"):
                output = self.linear_proj4(tf.concat([cell_output, context_vector], axis=1))
            outputs.append(output)

        # If using coverage, reshape it
        if coverage is not None:
            coverage = tf.reshape(coverage, [batch_size, -1])

        # return outputs, state, attn_dists, p_gens, coverage
        inputs["dec_outputs"] = outputs
        inputs["state"] = state
        inputs["attn_dists"] = attn_dists
        inputs["p_gens"] = p_gens
        inputs["coverage"] = coverage
        return inputs