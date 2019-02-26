import tensorflow as tf
import parameters
from estimator_config import estimator_cfg
from input_manager import InputFunction
from model import model_fn
import time


def load_ckpt(saver, sess, model_dir):
    """Load checkpoint from the ckpt_dir (if unspecified, this is train dir) and restore it to saver and sess, waiting 10 secs in the case of failure. Also returns checkpoint name."""
    while True:
        try:
            ckpt_dir = model_dir
            ckpt_state = tf.train.get_checkpoint_state(ckpt_dir)
            tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
            saver.restore(sess, ckpt_state.model_checkpoint_path)
            return ckpt_state.model_checkpoint_path
        except:
            tf.logging.info("Failed to load checkpoint from %s. Sleeping for %i secs...", ckpt_dir, 10)
            time.sleep(10)


def convert_to_coverage_model(hps, sess_cfg):
    """Load non-coverage checkpoint, add initialized extra variables for coverage, and save as new checkpoint"""
    tf.logging.info("converting non-coverage model to coverage model..")

    # initialize an entire coverage model from scratch
    sess = tf.Session(config=sess_cfg)
    print("initializing everything...")
    sess.run(tf.global_variables_initializer())

    # load all non-coverage weights from checkpoint
    saver = tf.train.Saver([v for v in tf.global_variables() if "coverage" not in v.name and "Adagrad" not in v.name])
    print("restoring non-coverage variables...")
    curr_ckpt = load_ckpt(saver, sess, hps.model_dir)
    print("restored.")

    # save this model and quit
    new_fname = curr_ckpt + '_cov_init'
    print("saving model to %s..." % (new_fname))
    new_saver = tf.train.Saver() # this one will save all variables that now exist
    new_saver.save(sess, new_fname)
    print("saved.")
    exit()


def main(_):
    hps = parameters.get_hps()
    tf.logging.info(hps)
    estimator = estimator_cfg(hps, model_fn=model_fn)

    if hps.convert_to_coverage_model == True and hps.coverage == True:
        # 无覆盖机制转为有覆盖机制，模型需要增加一些variable。
        # 这里的代码用于load无覆盖机制的模型，转为有覆盖机制的模型并save。

        input_wrapper = InputFunction(hps)
        inputs_dataset = input_wrapper.input_fn(tf.estimator.ModeKeys.TRAIN)
        iterator = inputs_dataset.make_initializable_iterator()
        features, label = iterator.get_next()

        tf.train.create_global_step()
        model_fn(features=features, labels=label, mode=tf.estimator.ModeKeys.TRAIN, params=hps)

        convert_to_coverage_model(hps, estimator.config.session_config)
        exit(0)

    elif hps.mode == tf.estimator.ModeKeys.TRAIN:
        input_wrapper = InputFunction(hps)
        estimator.train(lambda : input_wrapper.input_fn(tf.estimator.ModeKeys.TRAIN), steps=hps.run_step)

    elif hps.mode == tf.estimator.ModeKeys.PREDICT:
        decode_model_hps = hps._replace(max_dec_steps=1)
        estimator.predict(None)


if __name__ == "__main__":

    tf.logging.set_verbosity(tf.logging.INFO)

    tf.flags.FLAGS.data_dir = "F:/chinese_summarization"
    tf.flags.FLAGS.max_enc_steps = 50
    tf.flags.FLAGS.max_dec_steps = 10
    tf.flags.FLAGS.model_dir = "../model/"
    tf.flags.FLAGS.coverage = True
    # tf.flags.FLAGS.convert_to_coverage_model = True
    # tf.enable_eager_execution()

    tf.app.run(main)

