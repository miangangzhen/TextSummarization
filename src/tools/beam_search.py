import tensorflow as tf
import numpy as np
from tools.hyposis import Hypothesis

FLAGS = tf.flags.FLAGS

def run_beam_search(sess, model, vocab, batch):
    """Performs beam search decoding on the given example.

    Args:
      sess: a tf.Session
      model: a seq2seq model
      vocab: Vocabulary object
      batch: Batch object that is the same example repeated across the batch

    Returns:
      best_hyp: Hypothesis object; the best hypothesis found by beam search.
    """
    # Run the encoder to get the encoder hidden states and decoder initial state
    enc_states, dec_in_state = model.run_encoder(sess, batch)
    # dec_in_state is a LSTMStateTuple
    # enc_states has shape [batch_size, <=max_enc_steps, 2*hidden_dim].

    # Initialize beam_size-many hyptheses
    hyps = [Hypothesis(tokens=[vocab.word2id(data.START_DECODING)],
                       log_probs=[0.0],
                       state=dec_in_state,
                       attn_dists=[],
                       p_gens=[],
                       coverage=np.zeros([batch.enc_batch.shape[1]]) # zero vector of length attention_length
                       ) for _ in range(FLAGS.beam_size)]
    results = [] # this will contain finished hypotheses (those that have emitted the [STOP] token)

    steps = 0
    while steps < FLAGS.max_dec_steps and len(results) < FLAGS.beam_size:
        latest_tokens = [h.latest_token for h in hyps] # latest token produced by each hypothesis
        latest_tokens = [t if t in range(vocab.size()) else vocab.word2id(data.UNKNOWN_TOKEN) for t in latest_tokens] # change any in-article temporary OOV ids to [UNK] id, so that we can lookup word embeddings
        states = [h.state for h in hyps] # list of current decoder states of the hypotheses
        prev_coverage = [h.coverage for h in hyps] # list of coverage vectors (or None)

        # Run one step of the decoder to get the new info
        (topk_ids, topk_log_probs, new_states, attn_dists, p_gens, new_coverage) = model.decode_onestep(sess=sess,
                                                                                                        batch=batch,
                                                                                                        latest_tokens=latest_tokens,
                                                                                                        enc_states=enc_states,
                                                                                                        dec_init_states=states,
                                                                                                        prev_coverage=prev_coverage)

        # Extend each hypothesis and collect them all in all_hyps
        all_hyps = []
        num_orig_hyps = 1 if steps == 0 else len(hyps) # On the first step, we only had one original hypothesis (the initial hypothesis). On subsequent steps, all original hypotheses are distinct.
        for i in range(num_orig_hyps):
            h, new_state, attn_dist, p_gen, new_coverage_i = hyps[i], new_states[i], attn_dists[i], p_gens[i], new_coverage[i]  # take the ith hypothesis and new decoder state info
            for j in range(FLAGS.beam_size * 2):  # for each of the top 2*beam_size hyps:
                # Extend the ith hypothesis with the jth option
                new_hyp = h.extend(token=topk_ids[i, j],
                                   log_prob=topk_log_probs[i, j],
                                   state=new_state,
                                   attn_dist=attn_dist,
                                   p_gen=p_gen,
                                   coverage=new_coverage_i)
                all_hyps.append(new_hyp)

        # Filter and collect any hypotheses that have produced the end token.
        hyps = [] # will contain hypotheses for the next step
        for h in sort_hyps(all_hyps): # in order of most likely h
            if h.latest_token == vocab.word2id(data.STOP_DECODING): # if stop token is reached...
                # If this hypothesis is sufficiently long, put in results. Otherwise discard.
                if steps >= FLAGS.min_dec_steps:
                    results.append(h)
            else: # hasn't reached stop token, so continue to extend this hypothesis
                hyps.append(h)
            if len(hyps) == FLAGS.beam_size or len(results) == FLAGS.beam_size:
                # Once we've collected beam_size-many hypotheses for the next step, or beam_size-many complete hypotheses, stop.
                break

        steps += 1

    # At this point, either we've got beam_size results, or we've reached maximum decoder steps

    if len(results)==0: # if we don't have any complete results, add all current hypotheses (incomplete summaries) to results
        results = hyps

    # Sort hypotheses by average log probability
    hyps_sorted = sort_hyps(results)

    # Return the hypothesis with highest average log prob
    return hyps_sorted[0]

def sort_hyps(hyps):
    """Return a list of Hypothesis objects, sorted by descending average log probability"""
    return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)
