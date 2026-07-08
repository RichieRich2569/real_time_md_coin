function p = responsibilities_vector(obj)
%RESPONSIBILITIES_VECTOR Posterior per-context responsibilities (row vector).
%
%   p = responsibilities(obj) returns a 1-by-(max_contexts+1) row vector of the
%   posterior (post-observation) context probabilities for the current trial,
%   averaged over particles and mapped into the aligned global-context frame.
%   The trailing entry is the novel-context responsibility. This triggers (and
%   caches) the lazy context alignment.
%
%   This returns a VECTOR; RESPONSIBILITIES_MAP returns the same weights as a
%   containers.Map keyed by global context label.
%   PREDICTED_CONTEXT_PROBABILITIES_VECTOR is the prior (pre-observation)
%   counterpart.
%
%   See also RESPONSIBILITIES_MAP, PREDICTED_CONTEXT_PROBABILITIES_VECTOR,
%   CONTEXT_RESPONSIBILITIES_LOCAL, CONTEXT_ALIGNMENT.
    arguments
        obj (1, 1) RealTimeCOIN
    end
    p = contextProbabilityVector(obj, "responsibilities");
end
