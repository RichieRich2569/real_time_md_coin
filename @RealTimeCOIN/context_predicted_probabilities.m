function probs = context_predicted_probabilities(obj)
%CONTEXT_PREDICTED_PROBABILITIES Predicted context probabilities as a map.
%
%   probs = context_predicted_probabilities(obj) returns a containers.Map keyed
%   by aligned global context label (double) whose values are the prior
%   (pre-observation) context probabilities for the current trial. Only contexts
%   with strictly positive probability are included as keys. This triggers (and
%   caches) the lazy context alignment.
%
%   Naming hazard: this is the MAP form; the near-identically named
%   PREDICTED_CONTEXT_PROBABILITIES returns the same weights as a plain row
%   vector. See CONTEXT_RESPONSIBILITIES for the posterior (map) counterpart.
%
%   See also PREDICTED_CONTEXT_PROBABILITIES, CONTEXT_RESPONSIBILITIES,
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
