import os
from collections import namedtuple
import tensorflow as tf

# parameters
# Where to find data
from common.file_size import get_file_size

tf.flags.DEFINE_string('data_dir', '', '数据目录，应包含train/eval/predict三个子目录，及vocab文件')

# Important settings
tf.flags.DEFINE_string('mode', 'train', 'train/eval/infer')
tf.flags.DEFINE_string("model_dir", "", "estimator的model_dir")

# Hyperparameters
tf.flags.DEFINE_integer('hidden_dim', 256, 'dimension of RNN hidden states')
tf.flags.DEFINE_integer('embedding_size', 128, 'dimension of word embeddings')
tf.flags.DEFINE_integer('batch_size', 16, 'minibatch size')
tf.flags.DEFINE_integer('max_enc_steps', 400, 'max timesteps of encoder (max source text tokens)')
tf.flags.DEFINE_integer('max_dec_steps', 100, 'max timesteps of decoder (max summary tokens)')
tf.flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
tf.flags.DEFINE_integer('min_dec_steps', 35, 'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
tf.flags.DEFINE_integer('vocab_size', 50000, 'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')
tf.flags.DEFINE_float('lr', 0.15, 'learning rate')
tf.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
tf.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization')
tf.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')
tf.flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')
tf.flags.DEFINE_float("dropout", 0.1, "drop out prob")

# Pointer-generator or baseline model
tf.flags.DEFINE_boolean('pointer_gen', True, 'If True, use pointer-generator model. If False, use baseline model.')

# Coverage hyperparameters
tf.flags.DEFINE_boolean('coverage', False, 'Use coverage mechanism. Note, the experiments reported in the ACL paper train WITHOUT coverage until converged, and then train for a short phase WITH coverage afterwards. i.e. to reproduce the results in the ACL paper, turn this off for most of training then turn on for a short phase at the end.')
tf.flags.DEFINE_float('cov_loss_wt', 1.0, 'Weight of coverage loss (lambda in the paper). If zero, then no incentive to minimize coverage loss.')

tf.flags.DEFINE_integer('run_step', 8000, 'how many step run')
tf.flags.DEFINE_integer('check_steps', 1000, 'checkpoint保存频次')
tf.flags.DEFINE_string("embedding_file", "", "预训练的词向量文件")
tf.flags.DEFINE_boolean('convert_to_coverage_model', False, 'Convert a non-coverage model to a coverage model. Turn this on and run in train mode. Your current training model will be copied to a new version (same name with _cov_init appended) that will be ready to run with coverage flag turned on, for the coverage training stage.')


def get_hps():
    FLAGS = tf.flags.FLAGS
    hps_dict = {}
    for key, val in FLAGS.__flags.items():
        hps_dict[key] = val._value

    hps_dict["feature_voc_file"] = os.path.join(hps_dict["data_dir"], "vocab.txt")
    assert tf.gfile.Exists(hps_dict["feature_voc_file"]), "Vocab_file doesn't exist: {}".format(hps_dict["feature_voc_file"])

    # <pad>, <start>, <end>, vocabs..., <unk>
    hps_dict["feature_voc_file_len"] = 4 + get_file_size(hps_dict["feature_voc_file"])

    hps_dict["enc_input_name"] = "enc_input"
    hps_dict["dec_input_name"] = "dec_input"

    # in decode mode, run only one step
    if hps_dict["mode"]  == tf.estimator.ModeKeys.PREDICT:
        hps_dict["max_dec_steps"] = 1

    return namedtuple("HParams", list(hps_dict.keys()))(**hps_dict)

