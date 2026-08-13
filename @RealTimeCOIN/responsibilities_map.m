function probs = responsibilities_map(obj)
%RESPONSIBILITIES_MAP Posterior context responsibilities as a containers.Map.
%
%   probs = context_responsibilities(obj) returns a containers.Map keyed by
%   aligned global context label (double) whose values are the posterior
%   (post-observation) context probabilities for the current trial. Only contexts
%   with strictly positive probability are included as keys. This triggers (and
%   caches) the lazy context alignment.
%
%   This is the MAP form; RESPONSIBILITIES_VECTOR returns the same weights as a
%   plain row vector. PREDICTED_CONTEXT_PROBABILITIES_MAP is the prior (map)
%   counterpart.
%
%   See also RESPONSIBILITIES_VECTOR, PREDICTED_CONTEXT_PROBABILITIES_MAP,
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
