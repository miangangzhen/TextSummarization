import tensorflow as tf
from tensorflow.contrib.estimator import multi_label_head
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.python.estimator.run_config import RunConfig

from networks.embedding import EmbeddingLayer
from networks.encoder import EncoderLayer


def model_fn(features, labels, mode:tf.estimator.ModeKeys, params):
    tf.logging.info('Building graph...')

    with tf.variable_scope("rlex", initializer=xavier_initializer()) as scope:

        # # encoder part
        # enc_batch = features.get("enc_batch")
        # enc_lens = features.get("enc_lens")
        # enc_padding_mask = features.get("enc_padding_mask")
        # if params.pointer_gen:
        #     enc_batch_extend_vocab = features.get("enc_batch_extend_vocab")
        #     max_art_oovs = features.get("max_art_oovs")
        #
        # # decoder part
        # dec_batch = features.get("dec_batch")
        # target_batch = labels
        # dec_padding_mask = features.get("dec_padding_mask")
        # if mode == tf.estimator.ModeKeys.PREDICT and params.coverage:
        #     prev_coverage = features.get("prev_coverage")

        layers = []
        layers.append(EmbeddingLayer(params))
        layers.append(EncoderLayer(params))

        logits = features
        for layer in layers:
            logits = layer(logits)

        for key, value in logits.items():
            print(key)
            if isinstance(value, list):
                print("list of: {}".format(value[0].shape))
            elif isinstance(value, tf.nn.rnn_cell.LSTMStateTuple):
                print(value)
            else:
                print(value.shape)

        def train_op_fn(loss):
            train_op = tf.train.AdamOptimizer(learning_rate=params.learning_rate) \
                .minimize(loss, global_step=tf.train.get_global_step())
            return train_op
        head = multi_label_head(params.nClasses, weight_column=params.weightColName, thresholds=[0.3, 0.5, 0.6, 0.7])
        spec = head.create_estimator_spec(
            features, mode, logits, labels=labels, train_op_fn=train_op_fn)
        return spec

