function p = predicted_context_probabilities_vector(obj)
%PREDICTED_CONTEXT_PROBABILITIES_VECTOR Cross-run averaged prior context probs.
%   p = predicted_context_probabilities_vector(obj) returns the
%   1-by-(max_contexts+1) prior (pre-observation) context probability row,
%   averaged across runs in the common reference frame (docs/SPEC_ensemble.md
%   Part 10). Real contexts occupy slots 1..Kref, the novel-context probability
%   sits in slot Kref+1, and the row sums to 1. Read-only.
%
%   See also RESPONSIBILITIES_VECTOR, SAMPLED_CONTEXT_COUNT.
    arguments
        obj (1, 1) RealTimeCOINEnsemble
    end
    p = ensembleContextVector(obj, @(m) m.predicted_context_probabilities_vector());
end
