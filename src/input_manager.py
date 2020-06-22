import json
import os
import tensorflow as tf


class InputFunction(object):
    def __init__(self, params):
        self.params = params
        self.vocab = {}
        self._id_to_word = {}
        for i, word in enumerate(["<pad>", "<start>", "<end>"] + tf.gfile.GFile(self.params.feature_voc_file).readlines() + ["<unk>"]):
            word = word.strip()
            self.vocab[word] = i
            self._id_to_word[i] = word
        self.padded_shapes = ({"enc_input": [None],
                               "dec_input":[self.params.max_dec_steps],
                               "enc_len": [],
                               "dec_len":[],
                               "enc_input_extend_vocab": [None],
                               "article_oovs": [None]
                               }, [self.params.max_dec_steps])
        self.padded_values = ({"enc_input": 0,
                               "dec_input": 0,
                               "enc_len": 0,
                               "dec_len": 0,
                               "enc_input_extend_vocab": 0,
                               "article_oovs": ""}, 0)

    def get_data_dir(self, mode: tf.estimator.ModeKeys):
        if mode == tf.estimator.ModeKeys.TRAIN:
            return os.path.join(self.params.data_dir, "train")
        elif mode == tf.estimator.ModeKeys.EVAL:
            return os.path.join(self.params.data_dir, "eval")
        elif mode == tf.estimator.ModeKeys.PREDICT:
            return os.path.join(self.params.data_dir, "infer")
        else:
            raise ValueError("mode must be train/eval/infer")

    def prepare_data_for_train(self, data_set):
        return data_set.map(lambda x: tf.py_func(self.parse, [x], Tout=[
            tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.string])) \
            .map(lambda enc_id, dec_id, tar_id, enc_len, dec_len, enc_input_extend_vocab, article_oovs: ({
                                                                                                             "enc_input": tf.to_int32(enc_id),
                                                                                                             "dec_input": tf.to_int32(dec_id),
                                                                                                             "enc_len": tf.to_int32(enc_len),
                                                                                                             "dec_len": tf.to_int32(dec_len),
                                                                                                             "enc_input_extend_vocab": tf.to_int32(enc_input_extend_vocab),
                                                                                                             "article_oovs": article_oovs},
                                                                                                         tf.to_int32(tar_id))) \
            .padded_batch(self.params.batch_size,
                          padded_shapes=self.padded_shapes,
                          padding_values=self.padded_values).prefetch(self.params.batch_size * 10)

    def prepare_data_for_predict(self, data_set):
        padded_shapes = self.padded_shapes[0]
        padded_shapes.pop("dec_input")
        padded_shapes.pop("dec_len")

        padded_values = self.padded_values[0]
        padded_values.pop("dec_input")
        padded_values.pop("dec_len")

        return data_set.map(lambda x: tf.py_func(self.parse_predict, [x], Tout=[
            tf.int64, tf.int64, tf.int64, tf.string])).map(
            lambda enc_id, enc_len, enc_input_extend_vocab, article_oovs:{
                    "enc_input": tf.to_int32(enc_id),
                    "enc_len": tf.to_int32(enc_len),
                    "enc_input_extend_vocab": tf.to_int32(enc_input_extend_vocab),
                    "article_oovs": article_oovs
            })

    def input_fn(self, mode: tf.estimator.ModeKeys, limit=None):
        data_dir = self.get_data_dir(mode)
        file_paths = tf.gfile.Glob(os.path.join(data_dir, "*.json"))
        data_set = tf.data.TextLineDataset(file_paths, buffer_size=10 * 1024 * 1024)

        if mode == tf.estimator.ModeKeys.TRAIN:
            data_set = data_set.repeat(None).shuffle(buffer_size=20 * 1000)
            data_set = self.prepare_data_for_train(data_set)

        elif mode == tf.estimator.ModeKeys.EVAL:
            data_set = data_set.repeat(1)
            data_set = self.prepare_data_for_train(data_set)
        else:
            limit = -1 if limit == None else limit
            data_set = data_set.repeat(1).take(limit)
            data_set = self.prepare_data_for_predict(data_set)


        return data_set

    def parse(self, raw):
        d = json.loads(raw.decode('utf-8'))

        enc = d["content"].replace("\n", "").split(" ")[:self.params.max_enc_steps]
        enc_id = [self.word2id(x) for x in enc]
        enc_len = len(enc_id)

        title = d["title"].replace("\n", "").split(" ")[:self.params.max_dec_steps]
        title_id = [self.word2id(x) for x in title]
        dec_id = [self.word2id("<start>")] + title_id[:self.params.max_dec_steps - 1]
        dec_len = len(dec_id)

        tar_id = title_id \
            if len(title_id) == self.params.max_dec_steps \
            else title_id + [self.word2id("<end>")]

        # If using pointer-generator mode, we need to store some extra info
        enc_input_extend_vocab = []
        article_oovs = []
        if self.params.pointer_gen:
            enc_input_extend_vocab, article_oovs = self.content2id(enc)
            abs_ids_extend_vocab = self.title2id(title, article_oovs)
            tar_id = abs_ids_extend_vocab \
                if len(abs_ids_extend_vocab) == self.params.max_dec_steps \
                else abs_ids_extend_vocab + [self.word2id("<end>")]

        article_oovs = article_oovs if len(article_oovs) else [""]

        return enc_id, dec_id, tar_id, enc_len, dec_len, enc_input_extend_vocab, article_oovs

    def parse_predict(self, raw):
        d = json.loads(raw.decode("utf-8"))

        enc = d["content"].replace("\n", "").split(" ")[:self.params.max_enc_steps]
        tf.logging.info("origin article: {}".format("".join(enc)))
        enc_id = [self.word2id(x) for x in enc]
        enc_len = len(enc_id)

        # If using pointer-generator mode, we need to store some extra info
        enc_input_extend_vocab = []
        article_oovs = []
        if self.params.pointer_gen:
            enc_input_extend_vocab, article_oovs = self.content2id(enc)

        article_oovs = article_oovs if len(article_oovs) else [""]

        return enc_id, enc_len, enc_input_extend_vocab, article_oovs

    def title2id(self, title, article_oovs):
        ids = []
        unk_id = self.vocab.get("<unk>")
        for w in title:
            i = self.word2id(w)
            if i == unk_id:  # If w is an OOV word
                if w in article_oovs:  # If w is an in-article OOV
                    vocab_idx = len(self.vocab) + article_oovs.index(w)  # Map to its temporary article OOV number
                    ids.append(vocab_idx)
                else:  # If w is an out-of-article OOV
                    ids.append(unk_id)  # Map to the UNK token id
            else:
                ids.append(i)
        return ids

    def content2id(self, article):
        ids = []
        oovs = []
        unk_id = self.vocab.get("<unk>")
        for w in article:
            i = self.word2id(w)
            if i == unk_id:  # If w is OOV
                if w not in oovs:  # Add to list of OOVs
                    oovs.append(w)
                oov_num = oovs.index(w)  # This is 0 for the first article OOV, 1 for the second article OOV...
                ids.append(
                    len(self.vocab) + oov_num)  # This is e.g. 50000 for the first article OOV, 50001 for the second...
            else:
                ids.append(i)
        return ids, oovs

    def word2id(self, word):
        # id 0 for unknown word
        id = self.vocab.get(word, None)
        if not id:
            id = self.vocab.get("<unk>")
        return id

    def outputids2words(self, id_list, article_oovs):
        end_id = self.vocab.get("<end>")
        words = []
        for i in id_list:
            if i == end_id:
                break
            w = self._id_to_word.get(i, None)
            if w is None:
                article_oov_idx = i - len(self.vocab)
                w = "__" + article_oovs[article_oov_idx].decode("utf-8") + "__"
            words.append(w)
        return words
