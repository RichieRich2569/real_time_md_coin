function weights = contextProbabilityVector(obj, kind)
    arguments
        obj (1,1) RealTimeCOIN
        kind {mustBeMember(kind, ["predicted", "responsibilities", "count"])} = "predicted"
    end
    alignment = obj.ensureContextAlignment();
    switch kind
        case "predicted"
            W = obj.globalContextWeights(obj.D.predicted_probabilities, alignment);
            weights = mean(W, 2)';
        case "responsibilities"
            W = obj.globalContextWeights(obj.D.responsibilities, alignment);
            weights = mean(W, 2)';
        case "count"
            c = obj.globalSampledContexts(alignment);
            weights = histcounts(c, 0.5:1:(obj.max_contexts+1.5)) ./ max(numel(c), 1);
    end
    weights(~isfinite(weights)) = 0;
    s = sum(weights);
    if s > 0
        weights = weights ./ s;
    end
end
