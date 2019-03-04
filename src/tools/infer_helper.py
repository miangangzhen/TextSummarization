import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import xavier_initializer

from networks.decoder import DecoderLayer
from networks.embedding import EmbeddingLayer
from networks.projection import ProjectionLayer


class InferHelper(object):
    def __init__(self, encode_result, hps):
        self.enc_states = np.tile(encode_result["enc_states"], [hps.batch_size, 1, 1])
        self.enc_padding_mask = np.tile(encode_result["enc_padding_mask"], [hps.batch_size, 1])
        self.beam_size = hps.batch_size
        self.hps = hps


        if hps.pointer_gen:
            self.enc_input_extend_vocab = np.tile(encode_result["enc_input_extend_vocab"], [hps.batch_size, 1])
            self.article_oovs = np.tile(encode_result["article_oovs"], [hps.batch_size, 1])


    def decode_onestep(self, latest_tokens, dec_init_states, prev_coverage, sess: tf.Session, infer_model):
        # Turn dec_init_states (a list of LSTMStateTuples) into a single LSTMStateTuple for the batch
        new_c = np.stack([state[0] for state in dec_init_states])
        new_h = np.stack([state[1] for state in dec_init_states])

        if self.hps.coverage:
            prev_coverage = np.stack(prev_coverage, axis=0)

        feed = {
            infer_model.enc_states: self.enc_states,
            infer_model.enc_mask: self.enc_padding_mask,
            infer_model.dec_in_state_c: new_c,
            infer_model.dec_in_state_h: new_h,
            infer_model.dec_batch: np.transpose(np.array([latest_tokens]))
        }
        if self.hps.pointer_gen:
            pass
            feed[infer_model.enc_input_extend_vocab] = self.enc_input_extend_vocab
            feed[infer_model.article_oovs] = self.article_oovs

        results = sess.run(infer_model.to_return, feed_dict=feed)
        # for key, value in results.items():
        #     print(key)
        #     if isinstance(value, tf.nn.rnn_cell.LSTMStateTuple):
        #         print("lstm tuple")
        #     elif isinstance(value, list):
        #         print("{} * {}".format(value[0].shape, len(value)))
        #     else:
        #         print(value.shape)
        # print("decode_onestep")
        # pass

        # Convert results['states'] (a single LSTMStateTuple) into a list of LSTMStateTuple -- one for each hypothesis
        new_states = [tf.contrib.rnn.LSTMStateTuple(results['states'].c[i, :], results['states'].h[i, :]) for i in
                      range(self.hps.batch_size)]

        # Convert singleton list containing a tensor to a list of k arrays
        assert len(results['attn_dists']) == 1
        attn_dists = results['attn_dists'][0].tolist()

        if self.hps.pointer_gen:
            # Convert singleton list containing a tensor to a list of k arrays
            assert len(results['p_gens'])==1
            p_gens = results['p_gens'][0].tolist()
        else:
            p_gens = [None for _ in range(self.hps.batch_size)]

        # Convert the coverage tensor to a list length k containing the coverage vector for each hypothesis
        if self.hps.coverage:
            new_coverage = results['coverage'].tolist()
            assert len(new_coverage) == self.hps.batch_size
        else:
            new_coverage = [None for _ in range(self.hps.batch_size)]

        return results['ids'], results['probs'], new_states, attn_dists, p_gens, new_coverage


class InferModel(object):
    def __init__(self, hps):
        self.hps = hps
        self.build()

    def _add_placeholder(self):
        self.dec_batch = tf.placeholder(tf.int32, [self.hps.batch_size, 1], name='dec_batch')
        self.enc_states = tf.placeholder(tf.float32, [self.hps.batch_size, None, self.hps.hidden_dim * 2], name="enc_states")
        self.enc_mask = tf.placeholder(tf.int32, [self.hps.batch_size, None], name="enc_mask")
        self.dec_in_state_c = tf.placeholder(tf.float32, [self.hps.batch_size, self.hps.hidden_dim], name="dec_in_state_c")
        self.dec_in_state_h = tf.placeholder(tf.float32, [self.hps.batch_size, self.hps.hidden_dim], name="dec_in_state_h")
        self.dec_in_state = tf.nn.rnn_cell.LSTMStateTuple(self.dec_in_state_c, self.dec_in_state_h)
        if self.hps.coverage:
            self.prev_coverage = tf.placeholder(tf.float32, [None])
        if self.hps.pointer_gen:
            self.enc_input_extend_vocab = tf.placeholder(tf.int32, [self.hps.batch_size, None])
            self.article_oovs = tf.placeholder(tf.string, [self.hps.batch_size, None])

    def build(self):
        self._add_placeholder()

        with tf.variable_scope("pointer_generator", initializer=xavier_initializer()) as scope:

            embedding_result = EmbeddingLayer(self.hps)({"dec_input": self.dec_batch})

            decoder_feed = {
                "enc_states": self.enc_states,
                "enc_mask": self.enc_mask,
                "dec_in_state": self.dec_in_state,
                "dec_input": embedding_result["dec_input"]
            }
            if self.hps.coverage:
                decoder_feed["coverage"] = self.prev_coverage
            decoder_result = DecoderLayer(self.hps, mode=tf.estimator.ModeKeys.PREDICT)(decoder_feed)

            decoder_result["article_oovs"] = self.article_oovs
            decoder_result["enc_input_extend_vocab"] = self.enc_input_extend_vocab

            project_result = ProjectionLayer(self.hps, mode=tf.estimator.ModeKeys.PREDICT)(decoder_result)

            self.to_return = {
                "ids": project_result["topk_ids"],
                "probs": project_result["topk_log_probs"],
                "states": decoder_result["state"],
                "attn_dists": project_result["attn_dists"]
            }

            if self.hps.pointer_gen:
                self.to_return['p_gens'] = project_result["p_gens"]

            if self.hps.coverage:
                self.to_return["coverage"] = project_result["coverage"]
