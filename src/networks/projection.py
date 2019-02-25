import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

class ProjectionLayer(tf.layers.Layer):
    def __init__(self, params, dtype=tf.float32, name="projection"):
        super(ProjectionLayer, self).__init__(True, name, dtype)
        self.params = params

    def build(self, _):
        self.proj = tf.layers.Dense(self.params.feature_voc_file_len,
                        None, True, xavier_initializer(),
                        tf.truncated_normal_initializer(stddev=self.params.trunc_norm_init_std))
        self.built = True

    def call(self, inputs, **kwargs):
        # vocab_scores is the vocabulary distribution before applying softmax. Each entry on the list corresponds to one decoder step
        vocab_scores = []
        for i, output in enumerate(inputs["dec_outputs"]):
            vocab_scores.append(self.proj(output))  # apply the linear layer

        # The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.
        vocab_dists = [tf.nn.softmax(s) for s in vocab_scores]

        # For pointer-generator model, calc final distribution from copy distribution and vocabulary distribution
        if self.params.pointer_gen:
            max_art_oovs = tf.shape(inputs["article_oovs"])[1]
            final_dists = self._calc_final_dist(vocab_dists, inputs["attn_dists"], inputs["p_gens"], max_art_oovs, inputs["enc_input_extend_vocab"])
        else:  # final distribution is just vocabulary distribution
            final_dists = vocab_dists

        inputs["final_dists"] = final_dists
        inputs["vocab_scores"] = vocab_scores
        return inputs

    def _calc_final_dist(self, vocab_dists, attn_dists, p_gens, max_art_oovs, enc_input_extend_vocab):
        """Calculate the final distribution, for the pointer-generator model

        Args:
          vocab_dists: The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.
          attn_dists: The attention distributions. List length max_dec_steps of (batch_size, attn_len) arrays.

        Returns:
          final_dists: The final distributions. List length max_dec_steps of (batch_size, extended_vsize) arrays.
        """
        with tf.variable_scope('final_distribution'):
            # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
            # shape = [batch_size, vocab_size] * max_dec_steps
            vocab_dists = [p_gen * dist for (p_gen,dist) in zip(p_gens, vocab_dists)]
            # shape = [batch_size, enc_len] * max_dec_steps
            attn_dists = [(1-p_gen) * dist for (p_gen,dist) in zip(p_gens, attn_dists)]

            # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
            # extended_vsize = vocab + max_art_oovs, the maximum (over the batch) size of the extended vocabulary
            extended_vsize = self.params.feature_voc_file_len + max_art_oovs
            extra_zeros = tf.zeros((self.params.batch_size, max_art_oovs))
            # list length max_dec_steps of shape (batch_size, extended_vsize)
            vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists]

            # Project the values in the attention distributions onto the appropriate entries in the final distributions
            # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary, then we add 0.1 onto the 500th entry of the final distribution
            # This is done for each decoder timestep.
            # This is fiddly; we use tf.scatter_nd to do the projection
            # shape (batch_size), [0, 1, ..., batch_size-1]
            batch_nums = tf.range(0, limit=self.params.batch_size)
            batch_nums = tf.expand_dims(batch_nums, 1) # shape (batch_size, 1)
            # attn_len == enc_len
            attn_len = tf.shape(enc_input_extend_vocab)[1] # number of states we attend over
            # [0, 0, ..., 0]
            # [1, 1, ..., 1]
            # [............]
            batch_nums = tf.tile(batch_nums, [1, attn_len]) # shape (batch_size, attn_len)
            # shape (batch_size, enc_len, 2)
            indices = tf.stack( (batch_nums, enc_input_extend_vocab), axis=2)
            # final_dist shape = [batch_size, vocab_size + oov_size]
            shape = [self.params.batch_size, extended_vsize]
            attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attn_dists] # list length max_dec_steps (batch_size, extended_vsize)

            # Add the vocab distributions and the copy distributions together to get the final distributions
            # final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving the final distribution for that decoder timestep
            # Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
            final_dists = [vocab_dist + copy_dist for (vocab_dist,copy_dist) in zip(vocab_dists_extended, attn_dists_projected)]

            return final_dists