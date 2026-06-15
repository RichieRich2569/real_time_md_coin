function probs = context_responsibilities(obj)
    weights = contextProbabilityVector(obj, "responsibilities");
    probs = containers.Map('KeyType', 'double', 'ValueType', 'double');
    for c = 1:numel(weights)
        if weights(c) > 0
            probs(c) = weights(c);
        end
    end
end
