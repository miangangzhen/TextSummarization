import tensorflow as tf


class LossLayer(tf.layers.Layer):
    def __init__(self, params, mode:tf.estimator.ModeKeys, dtype=tf.float32, name="loss"):
        super(LossLayer, self).__init__(True, name, dtype)
        self.params = params
        self.mode = mode

    def build(self, input_shape):

        self.built = True

    def call(self, inputs, **kwargs):
        if self.mode in ['train', 'eval']:
            if self.params.pointer_gen:
                # Calculate the loss per step
                # This is fiddly; we use tf.gather_nd to pick out the probabilities of the gold target words
                loss_per_step = []  # will be list length max_dec_steps containing shape (batch_size)
                batch_nums = tf.range(0, limit=self.params.batch_size) # shape (batch_size)
                for dec_step, dist in enumerate(inputs["final_dists"]):
                    targets = inputs["target"][:,dec_step]  # The indices of the target words. shape (batch_size)
                    indices = tf.stack( (batch_nums, targets), axis=1) # shape (batch_size, 2)
                    gold_probs = tf.gather_nd(dist, indices)  # shape (batch_size). prob of correct words on this step
                    losses = -tf.log(gold_probs)
                    loss_per_step.append(losses)

                # Apply dec_padding_mask and get loss
                inputs["loss"] = self._mask_and_avg(loss_per_step, inputs["dec_mask"])

            else:  # baseline model
                inputs["loss"] = tf.contrib.seq2seq.sequence_loss(tf.stack(inputs["vocab_scores"], axis=1), inputs["target"],
                                                         inputs["dec_mask"])  # this applies softmax internally

            tf.summary.scalar('loss', inputs["loss"])

            # Calculate coverage loss from the attention distributions
            if self.params.coverage:
                with tf.variable_scope('coverage_loss'):
                    inputs["_coverage_loss"] = self._coverage_loss(inputs["attn_dists"], inputs["dec_mask"])
                    tf.summary.scalar('coverage_loss', inputs["_coverage_loss"])
                    inputs["total_loss"] = inputs["loss"] + self.params.cov_loss_wt * inputs["_coverage_loss"]
                tf.summary.scalar('total_loss', inputs["total_loss"])

        return inputs

    def _mask_and_avg(self, values, padding_mask):
        """Applies mask to values then returns overall average (a scalar)

            Args:
              values: a list length max_dec_steps containing arrays shape (batch_size).
              padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.

            Returns:
              a scalar
            """

        dec_lens = tf.reduce_sum(padding_mask, axis=1)  # shape batch_size. float32
        values_per_step = [v * padding_mask[:, dec_step] for dec_step, v in enumerate(values)]
        values_per_ex = sum(values_per_step) / dec_lens  # shape (batch_size); normalized value for each batch member
        return tf.reduce_mean(values_per_ex)  # overall average

    def _coverage_loss(self, attn_dists, padding_mask):
        """Calculates the coverage loss from the attention distributions.

        Args:
          attn_dists: The attention distributions for each decoder timestep. A list length max_dec_steps containing shape (batch_size, attn_length)
          padding_mask: shape (batch_size, max_dec_steps).

        Returns:
          coverage_loss: scalar
        """
        coverage = tf.zeros_like(attn_dists[0])  # shape (batch_size, attn_length). Initial coverage is zero.
        covlosses = []  # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
        for a in attn_dists:
            covloss = tf.reduce_sum(tf.minimum(a, coverage), [1])  # calculate the coverage loss for this step
            covlosses.append(covloss)
            coverage += a  # update the coverage vector
        coverage_loss = self._mask_and_avg(covlosses, padding_mask)
        return coverage_loss
