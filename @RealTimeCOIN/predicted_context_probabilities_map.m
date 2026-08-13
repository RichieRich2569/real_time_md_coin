function probs = predicted_context_probabilities_map(obj)
%PREDICTED_CONTEXT_PROBABILITIES_MAP Predicted context probabilities as a map.
%
%   probs = context_predicted_probabilities(obj) returns a containers.Map keyed
%   by aligned global context label (double) whose values are the prior
%   (pre-observation) context probabilities for the current trial. Only contexts
%   with strictly positive probability are included as keys. This triggers (and
%   caches) the lazy context alignment.
%
%   This is the MAP form; PREDICTED_CONTEXT_PROBABILITIES_VECTOR returns the same
%   weights as a plain row vector. RESPONSIBILITIES_MAP is the posterior (map)
%   counterpart.
%
%   See also PREDICTED_CONTEXT_PROBABILITIES_VECTOR, RESPONSIBILITIES_MAP,
%   CONTEXT_ALIGNMENT.
    arguments
        obj (1, 1) RealTimeCOIN
    end
    weights = contextProbabilityVector(obj, "predicted");
    probs = containers.Map('KeyType', 'double', 'ValueType', 'double');
    for c = 1:numel(weights)
        if weights(c) > 0
            probs(c) = weights(c);
        end
    end
end
