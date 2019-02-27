#!-*-coding=utf-8-*-
import numpy as np
import tensorflow as tf
import parameters
from estimator_config import estimator_cfg
from input_manager import InputFunction
from model import model_fn
from tools.hyposis import Hypothesis
from tools.transform_to_coverage_model import transform_to_coverage_model

def main(_):
    hps = parameters.get_hps()
    tf.logging.info(hps)
    estimator = estimator_cfg(hps, model_fn=model_fn)

    if hps.convert_to_coverage_model == True and hps.coverage == True:
        # 无覆盖机制转为有覆盖机制，模型需要增加一些variable。
        # 这里的代码用于load无覆盖机制的模型，转为有覆盖机制的模型并save。
        transform_to_coverage_model(hps, estimator)
        exit(0)

    elif hps.mode == tf.estimator.ModeKeys.TRAIN:
        input_wrapper = InputFunction(hps)
        estimator.train(lambda : input_wrapper.input_fn(tf.estimator.ModeKeys.TRAIN), steps=hps.run_step)

    elif hps.mode == tf.estimator.ModeKeys.PREDICT:
        input_wrapper = InputFunction(hps)

        # import tensorflow.contrib.eager as tfe
        # dataset = input_wrapper.input_fn(tf.estimator.ModeKeys.PREDICT)
        # for one_element in tfe.Iterator(dataset):
        #     model_fn(one_element, None, mode=hps.mode, params=hps)
        #     for key, value in one_element.items():
        #         print(key)
        #         print(value.shape)
        #     break

        encode_results = []
        for encode_result in estimator.predict(lambda : input_wrapper.input_fn(tf.estimator.ModeKeys.PREDICT)):
            encode_results.append(encode_result)
            break

        for encode_result in encode_results:
            hyps = [Hypothesis(tokens=[input_wrapper.word2id("<start>")],
                               log_probs=[0.0],
                               state_c=encode_result["dec_in_state_c"],
                               state_h=encode_result["dec_in_state_h"],
                               attn_dists=[],
                               p_gens=[],
                               coverage=np.zeros([len(encode_result["enc_states"][0])])
                               # zero vector of length attention_length
                               ) for _ in range(hps.beam_size)]

            break


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

