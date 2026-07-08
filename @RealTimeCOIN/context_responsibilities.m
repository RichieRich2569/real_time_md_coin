function probs = context_responsibilities(obj)
%CONTEXT_RESPONSIBILITIES Posterior context responsibilities as a map.
%
%   probs = context_responsibilities(obj) returns a containers.Map keyed by
%   aligned global context label (double) whose values are the posterior
%   (post-observation) context probabilities for the current trial. Only contexts
%   with strictly positive probability are included as keys. This triggers (and
%   caches) the lazy context alignment.
%
%   This is the MAP form; RESPONSIBILITIES returns the same weights as a plain
%   row vector. CONTEXT_PREDICTED_PROBABILITIES is the prior (map) counterpart.
%
%   See also RESPONSIBILITIES, CONTEXT_PREDICTED_PROBABILITIES,
%   CONTEXT_ALIGNMENT.
    arguments
        obj (1, 1) RealTimeCOIN
    end
    weights = contextProbabilityVector(obj, "responsibilities");
    probs = containers.Map('KeyType', 'double', 'ValueType', 'double');
    for c = 1:numel(weights)
        if weights(c) > 0
            probs(c) = weights(c);
        end
    end
end
