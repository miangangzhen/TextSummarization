import tensorflow as tf
from tensorflow.python.estimator.run_config import RunConfig


def estimator_cfg(hps, model_fn):
    # estimator运行环境配置
    session_config = tf.ConfigProto()
    session_config.allow_soft_placement = True
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.9
    session_config.gpu_options.allow_growth = True

    config = RunConfig(save_checkpoints_steps=hps.check_steps, keep_checkpoint_max=2,
                       session_config=session_config)

    return tf.estimator.Estimator(model_fn=model_fn, model_dir=hps.model_dir, config=config, params=hps)
