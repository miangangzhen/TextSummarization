import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from networks.decoder import DecoderLayer
from networks.embedding import EmbeddingLayer
from networks.encoder import EncoderLayer
from networks.loss import LossLayer
from networks.projection import ProjectionLayer


def tile_for_predict(features):
    features["enc_input"] = tf.reshape(features["enc_input"], [1, -1])
    features["enc_len"] = tf.reshape(features["enc_len"], [-1])
    features["enc_input_extend_vocab"] = tf.reshape(features["enc_input_extend_vocab"], [1, -1])
    features["article_oovs"] = tf.reshape(features["article_oovs"], [1, -1])
    return features


def model_fn(features, labels, mode: tf.estimator.ModeKeys, params):
    tf.logging.info('Building graph...')
    if mode == tf.estimator.ModeKeys.PREDICT:
        features = tile_for_predict(features)

    with tf.variable_scope("pointer_generator", initializer=xavier_initializer()) as scope:

        layers = []
        layers.append(EmbeddingLayer(params))
        layers.append(EncoderLayer(params))
        if mode != tf.estimator.ModeKeys.PREDICT:
            layers.append(DecoderLayer(params, mode=mode))
            layers.append(ProjectionLayer(params))
            layers.append(LossLayer(params, mode=mode))

        features["target"] = labels
        logits = features
        for layer in layers:
            logits = layer(logits)

        # for key, value in logits.items():
        #     print(key)
        #     if isinstance(value, list):
        #         print("list of: {}".format(value[0].shape))
        #     elif isinstance(value, tf.nn.rnn_cell.LSTMStateTuple):
        #         print(value)
        #     elif value is None:
        #         print(None)
        #     else:
        #         print(value.shape)

        if mode == tf.estimator.ModeKeys.TRAIN:
            # loss of the model
            loss_to_minimize = logits.get("total_loss") if params.coverage else logits["loss"]
            tvars = tf.trainable_variables()
            gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

            # Clip the gradients
            grads, global_norm = tf.clip_by_global_norm(gradients, params.max_grad_norm)
            tf.summary.scalar('global_norm', global_norm)

            # Apply adagrad optimizer
            optimizer = tf.train.AdagradOptimizer(params.lr, initial_accumulator_value=params.adagrad_init_acc)
            with tf.device("/gpu:0"):
                train_op = optimizer.apply_gradients(list(zip(grads, tvars)), global_step=tf.train.get_global_step(),
                                                           name='train_step')

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss_to_minimize,
                train_op=train_op)

        elif mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={
                    "enc_states": logits["enc_states"],
                    "dec_in_state_c": logits["dec_in_state"].c,
                    "dec_in_state_h": logits["dec_in_state"].h
                }
            )
        else:
            return None
