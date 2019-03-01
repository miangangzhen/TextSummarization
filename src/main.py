#!-*-coding=utf-8-*-
import numpy as np
import tensorflow as tf
import parameters
from estimator_config import estimator_cfg
from input_manager import InputFunction
from model import model_fn
from tools.hyposis import Hypothesis
from tools.infer_helper import InferHelper, build_infer_model
from tools.transform_to_coverage_model import transform_to_coverage_model

def main(_):
    hps = parameters.get_hps()

    # in decode mode, run only one step
    hps_for_predict = None
    if hps.mode == tf.estimator.ModeKeys.PREDICT:
        hps_for_predict = hps
        hps = hps._replace(max_dec_steps=1)

    tf.logging.info(hps)
    estimator = estimator_cfg(hps, model_fn=model_fn)
    input_wrapper = InputFunction(hps)

    if hps.convert_to_coverage_model == True and hps.coverage == True:
        # 无覆盖机制转为有覆盖机制，模型需要增加一些variable。
        # 这里的代码用于load无覆盖机制的模型，转为有覆盖机制的模型并save。
        transform_to_coverage_model(hps, estimator)
        exit(0)

    elif hps.mode == tf.estimator.ModeKeys.TRAIN:
        estimator.train(lambda : input_wrapper.input_fn(tf.estimator.ModeKeys.TRAIN), steps=hps.run_step)

    elif hps.mode == tf.estimator.ModeKeys.PREDICT:
        # import tensorflow.contrib.eager as tfe
        # dataset = input_wrapper.input_fn(tf.estimator.ModeKeys.PREDICT)
        # for one_element in tfe.Iterator(dataset):
        #     model_fn(one_element, None, mode=hps.mode, params=hps)
        #     for key, value in one_element.items():
        #         print(key)
        #         print(value.shape)
        #     break
        vocab_size = len(input_wrapper.vocab)

        encode_results = []
        for encode_result in estimator.predict(lambda : input_wrapper.input_fn(tf.estimator.ModeKeys.PREDICT)):
            encode_results.append(encode_result)
            break

        # 用estimator不太合适，每次predict都需要重新加载模型。
        # 因此改用底层API
        infer_model = build_infer_model(hps_for_predict)
        saver = tf.train.Saver()

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        # Load an initial checkpoint to use for decoding
        ckpt_state = tf.train.get_checkpoint_state(hps.model_dir)
        tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
        saver.restore(sess, ckpt_state.model_checkpoint_path)

        for encode_result in encode_results:
            print(encode_result)
            hyps = [Hypothesis(tokens=[input_wrapper.word2id("<start>")],
                               log_probs=[0.0],
                               state_c=encode_result["dec_in_state_c"],
                               state_h=encode_result["dec_in_state_h"],
                               attn_dists=[],
                               p_gens=[],
                               coverage=np.zeros([len(encode_result["enc_states"][0])])
                               # zero vector of length attention_length
                               ) for _ in range(hps.batch_size)]

            results = []  # this will contain finished hypotheses (those that have emitted the [STOP] token)

            steps = 0
            inferhelper = InferHelper(encode_result, hps_for_predict)

            while steps < hps_for_predict.max_dec_steps and len(results) < hps.batch_size:
                # latest token produced by each hypothesis
                latest_tokens = [h.latest_token for h in hyps]
                # change any in-article temporary OOV ids to [UNK] id, so that we can lookup word embeddings
                latest_tokens = [t if t < vocab_size else input_wrapper.word2id("<unk>") for t in
                                 latest_tokens]
                # list of current decoder states of the hypotheses
                states = [(h.state_c, h.state_h) for h in hyps]
                # list of coverage vectors (or None)
                prev_coverage = [h.coverage for h in hyps]

                # Run one step of the decoder to get the new info
                inferhelper.decode_onestep(latest_tokens, states, prev_coverage, sess, infer_model)


                steps += 1


if __name__ == "__main__":

    tf.logging.set_verbosity(tf.logging.INFO)

    tf.flags.FLAGS.mode = "infer"
    tf.flags.FLAGS.batch_size = 4

    # tf.flags.FLAGS.mode = "train"

    tf.flags.FLAGS.data_dir = "F:/chinese_summarization"
    tf.flags.FLAGS.max_enc_steps = 50
    tf.flags.FLAGS.max_dec_steps = 10
    tf.flags.FLAGS.model_dir = "../model/"
    tf.flags.FLAGS.coverage = True

    # only switch model from non-coverage to coverage
    # tf.flags.FLAGS.convert_to_coverage_model = True


    # tf.enable_eager_execution()

    tf.app.run(main)

