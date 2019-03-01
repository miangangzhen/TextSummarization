import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import xavier_initializer

from networks.embedding import EmbeddingLayer


class InferHelper(object):
    def __init__(self, encode_result, hps):
        self.enc_states = np.tile(encode_result["enc_states"], [hps.batch_size, 1, 1])
        self.enc_padding_mask = np.tile(encode_result["enc_padding_mask"], [hps.batch_size, 1, 1])
        self.beam_size = hps.batch_size
        self.hps = hps


        if hps.pointer_gen:
            self.enc_input_extend_vocab = np.tile(encode_result["enc_input_extend_vocab"], [hps.batch_size, 1, 1])
            self.article_oovs = np.tile(encode_result["article_oovs"], [hps.batch_size, 1, 1])


    def decode_onestep(self, latest_tokens, dec_init_states, prev_coverage, sess: tf.Session, infer_model):
        # enc_states = encode_result["enc_states"]

        # Turn dec_init_states (a list of LSTMStateTuples) into a single LSTMStateTuple for the batch
        new_c = np.stack([state[0] for state in dec_init_states])
        new_h = np.stack([state[1] for state in dec_init_states])
        new_dec_in_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)

        # if self.hps.pointer_gen:
        #     to_return['p_gens'] = self.p_gens

        if self.hps.coverage:
            prev_coverage = np.stack(prev_coverage, axis=0)
            # to_return['coverage'] = self.coverage

        feed = {
            # "enc_states": self.enc_states,
            # "enc_mask": self.enc_padding_mask,
            # "dec_in_state": new_dec_in_state,
            infer_model.dec_batch: np.transpose(np.array([latest_tokens]))
        }
        if self.hps.pointer_gen:
            pass
            # feed["enc_input_extend_vocab"] = self.enc_input_extend_vocab
            # feed["article_oovs"] = self.article_oovs

        dec_embedding = sess.run(infer_model.dec_embedding, feed_dict=feed)
        print(len(dec_embedding))
        print(dec_embedding[0].shape)
        print("decode_onestep")
        pass


class build_infer_model(object):
    def __init__(self, hps):
        self.hps = hps
        self.build()

    def build(self):
        with tf.variable_scope("pointer_generator", initializer=xavier_initializer()) as scope:
            self.dec_batch = tf.placeholder(tf.int32, [self.hps.batch_size, 1], name='dec_batch')
            result = EmbeddingLayer(self.hps)({"dec_input": self.dec_batch})
            self.dec_embedding = result["dec_input"]