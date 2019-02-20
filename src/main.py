import tensorflow as tf
import parameters
from estimator_config import estimator_cfg
from input_manager import InputFunction
from model import model_fn


def main(_):
    hps = parameters.get_hps()
    tf.logging.info(hps)
    estimator = estimator_cfg(hps, model_fn=model_fn)

    input_wrapper = InputFunction(hps)

    # debug input function
    import tensorflow.contrib.eager as tfe
    for features, label in tfe.Iterator(input_wrapper.input_fn(tf.estimator.ModeKeys.TRAIN)):
        model_fn(features, label, tf.estimator.ModeKeys.TRAIN, hps)
        # print(features)
        pass
    # run train
    # estimator.train(lambda : input_wrapper.input_fn(tf.estimator.ModeKeys.TRAIN), steps=hps.run_step)


if __name__ == "__main__":

    tf.logging.set_verbosity(tf.logging.INFO)

    tf.flags.FLAGS.data_dir = "F:/chinese_summarization"
    tf.flags.FLAGS.max_dec_steps = 10
    tf.enable_eager_execution()

    tf.app.run(main)

