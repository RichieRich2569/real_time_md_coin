function n = sampled_context_count(obj)
%SAMPLED_CONTEXT_COUNT Sampled-context occupancy across particles (row vector).
%
%   n = sampled_context_count(obj) returns a 1-by-(max_contexts+1) row vector
%   giving the fraction of particles whose sampled context for the current trial
%   equals each aligned global context (the trailing entry is the novel context).
%   The counts are normalised to sum to one. This triggers (and caches) the lazy
%   context alignment.
%
%   See also SAMPLED_CONTEXT_COUNT_LOCAL, RESPONSIBILITIES,
%   PREDICTED_CONTEXT_PROBABILITIES, CONTEXT_ALIGNMENT.
    arguments
        obj (1, 1) RealTimeCOIN
    end
    n = contextProbabilityVector(obj, "count");
end
