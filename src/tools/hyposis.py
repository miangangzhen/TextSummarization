#!-*-coding=utf-8-*-


class Hypothesis(object):
    """Class to represent a hypothesis during beam search. Holds all the information needed for the hypothesis."""

    def __init__(self, tokens, log_probs, state_c, state_h, attn_dists, p_gens, coverage):
        """Hypothesis constructor.

        Args:
          tokens: List of integers. The ids of the tokens that form the summary so far.
          log_probs: List, same length as tokens, of floats, giving the log probabilities of the tokens so far.
          state: Current state of the decoder, a LSTMStateTuple.
          attn_dists: List, same length as tokens, of numpy arrays with shape (attn_length). These are the attention distributions so far.
          p_gens: List, same length as tokens, of floats, or None if not using pointer-generator model. The values of the generation probability so far.
          coverage: Numpy array of shape (attn_length), or None if not using coverage. The current coverage vector.
        """
        self.tokens = tokens
        self.log_probs = log_probs
        self.state_c = state_c
        self.state_h = state_h
        self.attn_dists = attn_dists
        self.p_gens = p_gens
        self.coverage = coverage

    def extend(self, token, log_prob, state, attn_dist, p_gen, coverage):
        """Return a NEW hypothesis, extended with the information from the latest step of beam search.

        Args:
          token: Integer. Latest token produced by beam search.
          log_prob: Float. Log prob of the latest token.
          state: Current decoder state, a LSTMStateTuple.
          attn_dist: Attention distribution from latest step. Numpy array shape (attn_length).
          p_gen: Generation probability on latest step. Float.
          coverage: Latest coverage vector. Numpy array shape (attn_length), or None if not using coverage.
        Returns:
          New Hypothesis for next step.
        """
        return Hypothesis(tokens = self.tokens + [token],
                          log_probs = self.log_probs + [log_prob],
                          state = state,
                          attn_dists = self.attn_dists + [attn_dist],
                          p_gens = self.p_gens + [p_gen],
                          coverage = coverage)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def log_prob(self):
        # the log probability of the hypothesis so far is the sum of the log probabilities of the tokens so far
        return sum(self.log_probs)

    @property
    def avg_log_prob(self):
        # normalize log probability by number of tokens (otherwise longer sequences always have lower probability)
        return self.log_prob / len(self.tokens)
