import json
import os
import tensorflow as tf


class InputFunction(object):
    def __init__(self, params):
        self.params = params

    def get_data_dir(self, mode: tf.estimator.ModeKeys):
        if mode == tf.estimator.ModeKeys.TRAIN:
            return os.path.join(self.params.data_dir, "train")
        elif mode == tf.estimator.ModeKeys.EVAL:
            return os.path.join(self.params.data_dir, "eval")
        elif mode == tf.estimator.ModeKeys.PREDICT:
            return os.path.join(self.params.data_dir, "infer")
        else:
            raise ValueError("mode must be train/eval/infer")

    def input_fn(self, mode: tf.estimator.ModeKeys):
        data_dir = self.get_data_dir(mode)
        file_paths = tf.gfile.Glob(os.path.join(data_dir, "*.json"))
        data_set = tf.data.TextLineDataset(file_paths, buffer_size=10 * 1024 * 1024)

        if mode == tf.estimator.ModeKeys.TRAIN:
            data_set = data_set.repeat(None).shuffle(buffer_size=20 * 1000) \
                .map(lambda x: tf.py_func(self.parse, [x], Tout=[tf.string, tf.string, tf.string])) \
                .map(lambda x, y, z: ({"enc_input": x, "dec_input": y}, z))\
                .padded_batch(self.params.batch_size,
                              padded_shapes=({"enc_input": [None], "dec_input":[self.params.max_dec_steps]}, [None]),
                              padding_values=({"enc_input": "<pad>", "dec_input": "<pad>"}, "<pad>"))

        elif mode == tf.estimator.ModeKeys.EVAL:
            data_set = data_set.repeat(1) \
                .map(lambda x: tf.py_func(self.parse, [x], Tout=[tf.string, tf.string, tf.string])) \
                .map(lambda x, y, z: ({"enc": x, "dec": y}, z)) \
                .padded_batch(self.params.batch_size,
                              padded_shapes=({"enc_input": [None], "dec_input": [self.params.max_dec_steps]}, [None]),
                              padding_values=({"enc_input": "<pad>", "dec_input": "<pad>"}, "<pad>"))

        return data_set

    def parse(self, raw):
        d = json.loads(raw.decode('utf-8'))
        enc = d["content"].replace("\n", "").split(" ")[:self.params.max_enc_steps]
        dec = d["title"].replace("\n", "").split(" ")
        tar = dec if len(dec) < self.params.max_dec_steps else dec[:self.params.max_dec_steps - 1] + ["<end>"]
        return enc, ["<start>"] + enc[:self.params.max_dec_steps - 1], tar
