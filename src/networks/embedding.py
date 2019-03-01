import tensorflow as tf
from tensorflow.python.ops.lookup_ops import index_table_from_file
from common.pretrain_w2v import WordEmbeddingInitializer


class EmbeddingLayer(tf.layers.Layer):
    def __init__(self, params, dtype=tf.float32, name="word_embedding"):
        super(EmbeddingLayer, self).__init__(True, name, dtype)
        self.params = params
        self.num_oov_buckets = 2000

    def build(self, _):
        padding = self.add_variable(
            name="padding"
            , shape=[1, self.params.embedding_size]
            , initializer=tf.zeros_initializer()
            , trainable=False
        )

        if not self.params.embedding_file:
            embedding_other = self.add_variable(
                name="embedding_voc"
                , shape=[self.params.feature_voc_file_len - 2, self.params.embedding_size]
                , initializer=tf.random_uniform_initializer(-1, 1)
            )

        else:
            embedding_other = self.add_variable(
                name="embedding_voc",
                shape=[self.params.feature_voc_file_len - 2, self.params.embedding_size],
                initializer=WordEmbeddingInitializer(
                    self.params.embedding_file, include_word=False, vector_length=self.params.embedding_size
                )
            )
        tf.add_to_collection("not_in_ema", embedding_other)
        embedding_oov = self.add_variable(
            name="embedding_oov"
            , shape=[self.num_oov_buckets, self.params.embedding_size]
            , initializer=tf.random_uniform_initializer(-1, 1)
        )
        tf.add_to_collection("not_in_ema", embedding_oov)
        self.embedding = tf.concat([padding, embedding_other, embedding_oov], axis=0)
        if tf.executing_eagerly():
            tmp = tf.slice(self.embedding, [0, 0], [4, -1])
            print("embedding is \r\n {}".format(tmp))

        self.built = True

    def call(self, inputs, **kwargs):
        enc_id = inputs.get(self.params.enc_input_name, None)
        if enc_id != None:
            inputs["enc_mask"] = tf.cast(tf.abs(tf.sign(enc_id)), tf.float32)
            enc_embedding = tf.nn.embedding_lookup(self.embedding, enc_id)
            inputs[self.params.enc_input_name] = enc_embedding

        dec_id = inputs.get(self.params.dec_input_name, None)
        if dec_id != None:
            inputs["dec_mask"] = tf.cast(tf.abs(tf.sign(dec_id)), tf.float32)
            dec_embedding_list = tf.unstack(tf.nn.embedding_lookup(self.embedding, dec_id), axis=1)
            inputs[self.params.dec_input_name] = dec_embedding_list

        return inputs
