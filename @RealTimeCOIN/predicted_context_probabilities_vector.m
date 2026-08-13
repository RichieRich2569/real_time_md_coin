function p = predicted_context_probabilities_vector(obj)
%PREDICTED_CONTEXT_PROBABILITIES_VECTOR Predicted per-context probs (row vector).
%
%   p = predicted_context_probabilities(obj) returns a 1-by-(max_contexts+1)
%   row vector of the prior (pre-observation) context probabilities for the
%   current trial, averaged over particles and mapped into the aligned
%   global-context frame. The trailing entry is the novel-context probability.
%   This triggers (and caches) the lazy context alignment.
%
%   This returns a VECTOR; PREDICTED_CONTEXT_PROBABILITIES_MAP returns the same
%   weights as a containers.Map keyed by global context label.
%   RESPONSIBILITIES_VECTOR is the posterior (post-observation) counterpart.
%
%   See also PREDICTED_CONTEXT_PROBABILITIES_MAP, RESPONSIBILITIES_VECTOR,
%   PREDICTED_CONTEXT_PROBABILITIES_LOCAL, CONTEXT_ALIGNMENT.
    arguments
        obj (1, 1) RealTimeCOIN
    end
    p = contextProbabilityVector(obj, "predicted");
end
