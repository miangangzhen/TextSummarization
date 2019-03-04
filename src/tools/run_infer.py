import tensorflow as tf
import numpy as np
from tools.hyposis import Hypothesis
from tools.infer_helper import InferModel, InferHelper


def sort_hyps(hyps):
    """Return a list of Hypothesis objects, sorted by descending average log probability"""
    return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)


def run_infer(input_wrapper, hps_for_predict, estimator):
    vocab_size = len(input_wrapper.vocab)

    # 用estimator不太合适，每次predict都需要重新加载模型。
    # 因此改用底层API
    infer_model = InferModel(hps_for_predict)
    saver = tf.train.Saver()

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    # Load an initial checkpoint to use for decoding
    ckpt_state = tf.train.get_checkpoint_state(hps_for_predict.model_dir)
    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    saver.restore(sess, ckpt_state.model_checkpoint_path)

    for encode_result in estimator.predict(lambda: input_wrapper.input_fn(tf.estimator.ModeKeys.PREDICT, limit=200)):
        hyps = [Hypothesis(tokens=[input_wrapper.word2id("<start>")],
                           log_probs=[0.0],
                           state_c=encode_result["dec_in_state_c"],
                           state_h=encode_result["dec_in_state_h"],
                           attn_dists=[],
                           p_gens=[],
                           coverage=np.zeros([len(encode_result["enc_states"][0])])
                           # zero vector of length attention_length
                           ) for _ in range(hps_for_predict.batch_size)]

        results = []  # this will contain finished hypotheses (those that have emitted the [STOP] token)

        steps = 0
        inferhelper = InferHelper(encode_result, hps_for_predict)

        while steps < hps_for_predict.max_dec_steps and len(results) < hps_for_predict.batch_size:
            # latest token produced by each hypothesis
            latest_tokens = [h.latest_token for h in hyps]
            # change any in-article temporary OOV ids to [UNK] id, so that we can lookup word embeddings
            latest_tokens = [t if t < vocab_size else input_wrapper.word2id("<unk>") for t in
                             latest_tokens]
            # list of current decoder states of the hypotheses
            states = [(h.state_c, h.state_h) for h in hyps]
            # list of coverage vectors (or None)
            prev_coverage = [h.coverage for h in hyps]

            # Run one step of the decoder to get the new info
            topk_ids, topk_log_probs, new_states, attn_dists, p_gens, new_coverage = inferhelper.decode_onestep(
                latest_tokens, states, prev_coverage, sess, infer_model)

            # Extend each hypothesis and collect them all in all_hyps
            all_hyps = []
            num_orig_hyps = 1 if steps == 0 else len(
                hyps)  # On the first step, we only had one original hypothesis (the initial hypothesis). On subsequent steps, all original hypotheses are distinct.
            for i in range(num_orig_hyps):
                h, new_state, attn_dist, p_gen, new_coverage_i = hyps[i], new_states[i], attn_dists[i], p_gens[i], \
                                                                 new_coverage[
                                                                     i]  # take the ith hypothesis and new decoder state info
                for j in range(hps_for_predict.batch_size * 2):  # for each of the top 2*beam_size hyps:
                    # Extend the ith hypothesis with the jth option
                    new_hyp = h.extend(token=topk_ids[i, j],
                                       log_prob=topk_log_probs[i, j],
                                       state=new_state,
                                       attn_dist=attn_dist,
                                       p_gen=p_gen,
                                       coverage=new_coverage_i)
                    all_hyps.append(new_hyp)

            # Filter and collect any hypotheses that have produced the end token.
            hyps = []  # will contain hypotheses for the next step
            for h in sort_hyps(all_hyps):  # in order of most likely h
                if h.latest_token == input_wrapper.word2id("<end>"):  # if stop token is reached...
                    # If this hypothesis is sufficiently long, put in results. Otherwise discard.
                    if steps >= hps_for_predict.min_dec_steps:
                        results.append(h)
                else:  # hasn't reached stop token, so continue to extend this hypothesis
                    hyps.append(h)
                if len(hyps) == hps_for_predict.batch_size or len(results) == hps_for_predict.batch_size:
                    # Once we've collected beam_size-many hypotheses for the next step, or beam_size-many complete hypotheses, stop.
                    break

            steps += 1

        # At this point, either we've got beam_size results, or we've reached maximum decoder steps
        if len(
                results) == 0:  # if we don't have any complete results, add all current hypotheses (incomplete summaries) to results
            results = hyps

        # Sort hypotheses by average log probability
        hyps_sorted = sort_hyps(results)

        # Return the hypothesis with highest average log prob
        best_hyp = hyps_sorted[0]
        output_ids = [int(t) for t in best_hyp.tokens[1:]]
        decoded_words = input_wrapper.outputids2words(output_ids, encode_result["article_oovs"])

        decoded_output = ' '.join(decoded_words)
        print(decoded_output)