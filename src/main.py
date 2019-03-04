#!-*-coding=utf-8-*-
import numpy as np
import tensorflow as tf
import parameters
from estimator_config import estimator_cfg
from input_manager import InputFunction
from model import model_fn
from tools.hyposis import Hypothesis
from tools.infer_helper import InferHelper, InferModel
from tools.run_infer import run_infer
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
        run_infer(input_wrapper, hps_for_predict, estimator)


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

