from input_manager import InputFunction
import tensorflow as tf
from model import model_fn
import time


def load_ckpt(saver, sess, model_dir):
    """Load checkpoint from the ckpt_dir (if unspecified, this is train dir) and restore it to saver and sess, waiting 10 secs in the case of failure. Also returns checkpoint name."""
    while True:
        try:
            ckpt_state = tf.train.get_checkpoint_state(model_dir)
            tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
            saver.restore(sess, ckpt_state.model_checkpoint_path)
            return ckpt_state.model_checkpoint_path
        except Exception as e:
            print(e)
            tf.logging.info("Failed to load checkpoint from %s. Sleeping for %i secs...", model_dir, 10)
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

def transform_to_coverage_model(hps, estimator):

    input_wrapper = InputFunction(hps)
    inputs_dataset = input_wrapper.input_fn(tf.estimator.ModeKeys.TRAIN)
    iterator = inputs_dataset.make_initializable_iterator()
    features, label = iterator.get_next()

    tf.train.create_global_step()
    model_fn(features=features, labels=label, mode=tf.estimator.ModeKeys.TRAIN, params=hps)

    convert_to_coverage_model(hps, estimator.config.session_config)
