function weights = contextProbabilityVector(obj, kind)
%CONTEXTPROBABILITYVECTOR Global-frame context probability vector.
%   weights = contextProbabilityVector(obj, kind) returns a normalised
%   probability over the global context slots, averaged across the modal
%   particles. It is the globally aligned counterpart of
%   localContextProbabilityVector.
%
%   kind selects the quantity summarised:
%     "predicted"        mean of the aligned predicted context probabilities
%                        (before the observation).
%     "responsibilities" mean of the aligned context responsibilities (after
%                        the observation).
%     "count"            empirical frequency of the sampled global contexts.
%
%   Output:
%     weights  1-by-(max_contexts+1) row summing to 1 (or all zero when no mass
%              exists). Non-finite entries are treated as zero.
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
            % Integer-centred bin edges 0.5, 1.5, ..., (Cmax+1)+0.5 so that
            % histcounts places each context label k in its own bin k. Cmax+1
            % here is max_contexts+1 (novel slot), giving max_contexts+1 bins.
            edges = 0.5:1:(obj.max_contexts + 1.5);
            weights = histcounts(c, edges) ./ max(numel(c), 1);
    end
    weights(~isfinite(weights)) = 0;
    s = sum(weights);
    if s > 0
        weights = weights ./ s;
    end
end
