import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

class ProjectionLayer(tf.layers.Layer):
    def __init__(self, params, dtype=tf.float32, name="word_embedding"):
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
            final_dists = self._calc_final_dist(vocab_dists, inputs["attn_dists"])
        else:  # final distribution is just vocabulary distribution
            final_dists = vocab_dists

        inputs["final_dists"] = final_dists
        return inputs