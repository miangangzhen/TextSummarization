import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer


class EncoderLayer(tf.layers.Layer):
    def __init__(self, params, dtype=tf.float32, name="encoder"):
        super(EncoderLayer, self).__init__(True, name, dtype)
        self.params = params

    def build(self, _):
        self.cell_fw = tf.nn.rnn_cell.LSTMCell(self.params.hidden_dim, initializer=xavier_initializer(), state_is_tuple=True)
        self.cell_bw = tf.nn.rnn_cell.LSTMCell(self.params.hidden_dim, initializer=xavier_initializer(), state_is_tuple=True)
        if self.trainable:
            self.cell_fw = tf.nn.rnn_cell.DropoutWrapper(self.cell_fw, input_keep_prob=1-self.params.dropout, output_keep_prob=1-self.params.dropout)
            self.cell_bw = tf.nn.rnn_cell.DropoutWrapper(self.cell_bw, input_keep_prob=1-self.params.dropout, output_keep_prob=1-self.params.dropout)

        # Define w and b to reduce the c and h dim from 2 * hidden_dim to hidden_dim
        self.reduce_c = tf.layers.Dense(self.params.hidden_dim, activation=tf.nn.relu, use_bias=True,
                                        kernel_initializer=xavier_initializer(), trainable=self.trainable)
        self.reduce_h = tf.layers.Dense(self.params.hidden_dim, activation=tf.nn.relu, use_bias=True,
                                        kernel_initializer=xavier_initializer(), trainable=self.trainable)
        if self.trainable:
            self.dropout = tf.layers.Dropout(rate=self.params.dropout)

        self.built = True

    def call(self, inputs, **kwargs):
        encoder_inputs = inputs[self.params.enc_input_name]
        enc_len = inputs["enc_len"]
        (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw, encoder_inputs,
                                                                            dtype=tf.float32, sequence_length=enc_len,
                                                                            swap_memory=True)

        inputs["enc_states"] = tf.concat(axis=2, values=encoder_outputs)

        old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c])  # Concatenation of fw and bw cell
        old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h])  # Concatenation of fw and bw state
        new_c = self.reduce_c(old_c)
        new_h = self.reduce_h(old_h)
        if self.trainable:
            new_c = self.dropout(new_c)
            new_h = self.dropout(new_h)
        inputs["dec_in_state"] = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)

        return inputs
