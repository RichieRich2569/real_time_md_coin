function p = predicted_context_probabilities(obj)
%PREDICTED_CONTEXT_PROBABILITIES Predicted per-context probabilities (row vector).
%
%   p = predicted_context_probabilities(obj) returns a 1-by-(max_contexts+1)
%   row vector of the prior (pre-observation) context probabilities for the
%   current trial, averaged over particles and mapped into the aligned
%   global-context frame. The trailing entry is the novel-context probability.
%   This triggers (and caches) the lazy context alignment.
%
%   Note on the confusingly similar name: this returns a VECTOR, whereas
%   CONTEXT_PREDICTED_PROBABILITIES returns a containers.Map keyed by global
%   context label. See CONTEXT_PREDICTED_PROBABILITIES for the map form and
%   RESPONSIBILITIES for the posterior (post-observation) counterpart.
%
%   See also CONTEXT_PREDICTED_PROBABILITIES, RESPONSIBILITIES,
%   PREDICTED_CONTEXT_PROBABILITIES_LOCAL, CONTEXT_ALIGNMENT.
    arguments
        obj (1, 1) RealTimeCOIN
    end
    p = contextProbabilityVector(obj, "predicted");
end
