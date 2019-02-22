import tensorflow as tf
from tensorflow.contrib.estimator import multi_label_head
from tensorflow.contrib.layers import xavier_initializer

from networks.decoder import DecoderLayer
from networks.embedding import EmbeddingLayer
from networks.encoder import EncoderLayer
from networks.loss import LossLayer
from networks.projection import ProjectionLayer


def model_fn(features, labels, mode:tf.estimator.ModeKeys, params):
    tf.logging.info('Building graph...')

    with tf.variable_scope("pointer_generator", initializer=xavier_initializer()) as scope:

        layers = []
        layers.append(EmbeddingLayer(params))
        layers.append(EncoderLayer(params))
        layers.append(DecoderLayer(params, mode=mode))
        layers.append(ProjectionLayer(params))
        layers.append(LossLayer(params, mode=mode))

        features["target"] = labels
        logits = features
        for layer in layers:
            logits = layer(logits)

        for key, value in logits.items():
            print(key)
            if isinstance(value, list):
                print("list of: {}".format(value[0].shape))
            elif isinstance(value, tf.nn.rnn_cell.LSTMStateTuple):
                print(value)
            elif value is None:
                print(None)
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

