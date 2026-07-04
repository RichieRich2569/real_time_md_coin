function T = local_transition_probabilities(obj)
%LOCAL_TRANSITION_PROBABILITIES Expected local context-transition matrix.
%
%   T = local_transition_probabilities(obj) returns the K-by-(K+1) expected
%   local transition probability matrix in the aligned global-context frame,
%   where K = context_alignment(obj).K. Row i is the transition distribution
%   out of global context i; columns 1..K are the known contexts and column
%   K+1 is the novel context. Mirrors COIN's plot_local_transition_probabilities
%   (a single trial). Dimension-independent, so identical in form for the scalar
%   and multi-dimensional models.
    arguments
        obj (1, 1) RealTimeCOIN
    end
    alignment = ensureContextAlignment(obj);
    K = alignment.K;
    T = alignment.global_contexts.transition_prob(1:K, 1:K+1);
end
