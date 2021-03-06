# TextSummarization
Using tensorflow estimator api implement pointer-generator (reference: https://github.com/abisee/pointer-generator)

usage:
```python
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
        # convert non-coverage to coverage model
        transform_to_coverage_model(hps, estimator)
        exit(0)

    elif hps.mode == tf.estimator.ModeKeys.TRAIN:
        estimator.train(lambda : input_wrapper.input_fn(tf.estimator.ModeKeys.TRAIN), steps=hps.run_step)

    elif hps.mode == tf.estimator.ModeKeys.PREDICT:
        run_infer(input_wrapper, hps_for_predict, estimator)
        
if __name__ == "__main__":
    
    # setting log level
    tf.logging.set_verbosity(tf.logging.INFO)

    # infer mode
    tf.flags.FLAGS.mode = "infer"
    tf.flags.FLAGS.batch_size = 4

    # train mode
    # tf.flags.FLAGS.mode = "train"

    # parameters
    tf.flags.FLAGS.data_dir = "F:/chinese_summarization"
    tf.flags.FLAGS.max_enc_steps = 50
    tf.flags.FLAGS.max_dec_steps = 10
    tf.flags.FLAGS.model_dir = "../model/"
    tf.flags.FLAGS.coverage = True

    # only switch model from non-coverage to coverage
    # tf.flags.FLAGS.convert_to_coverage_model = True

    tf.app.run(main)
```