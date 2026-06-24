function weights = localContextProbabilityVector(obj, kind)
    Cmax = obj.max_contexts + 1;
    [Km, ~, modalIdx] = obj.selectModalContexts();
    weights = zeros(1, Cmax);
    if isempty(modalIdx)
        return;
    end

    switch kind
        case "predicted"
            W = obj.D.predicted_probabilities(:, modalIdx);
            weights = mean(W, 2)';
        case "responsibilities"
            W = obj.D.responsibilities(:, modalIdx);
            weights = mean(W, 2)';
        case "count"
            c = obj.D.context(modalIdx);
            weights = histcounts(c, 0.5:1:(Cmax + 0.5)) ./ numel(c);
    end

    if Km >= obj.max_contexts
        weights(Km+1:end) = 0;
    elseif Km + 2 <= Cmax
        weights(Km+2:end) = 0;
    end

    weights(~isfinite(weights)) = 0;
    s = sum(weights);
    if s > 0
        weights = weights ./ s;
    end
end
