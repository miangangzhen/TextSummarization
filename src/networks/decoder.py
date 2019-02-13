import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer


class DecoderLayer(tf.layers.Layer):
    def __init__(self, params, dtype=tf.float32, name="encoder"):
        super(DecoderLayer, self).__init__(True, name, dtype)
        self.params = params

    def build(self, input_shape):
        pass

    def __call__(self, inputs, **kwargs):

        pass