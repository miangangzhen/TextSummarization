import tensorflow as tf


def get_file_size(file_name):
    lines = tf.gfile.Open(file_name).readlines()
    voc_size = len(lines)
    return voc_size
