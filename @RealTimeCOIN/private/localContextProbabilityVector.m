function weights = localContextProbabilityVector(obj, kind)
%LOCALCONTEXTPROBABILITYVECTOR Local-frame context probability vector.
%   weights = localContextProbabilityVector(obj, kind) returns a normalised
%   probability over the local context slots, averaged across the modal
%   particles, without aligning labels across particles. It is the raw,
%   per-particle-label counterpart of contextProbabilityVector.
%
%   kind selects the quantity summarised:
%     "predicted"        mean of the local predicted context probabilities.
%     "responsibilities" mean of the local context responsibilities.
%     "count"            empirical frequency of the sampled local contexts.
%
%   Output:
%     weights  1-by-(max_contexts+1) row summing to 1 (or all zero when no modal
%              particle exists). Slots beyond the modal cardinality are zeroed.
    Cmax = obj.max_contexts + 1;                     % context slots incl. novel (+1)
    [Km, ~, modalIdx] = obj.selectModalContexts();   % modal cardinality and particles
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
            % Integer-centred bin edges 0.5, 1.5, ..., Cmax+0.5 so that
            % histcounts places each context label k in its own bin k.
            edges = 0.5:1:(Cmax + 0.5);
            weights = histcounts(c, edges) ./ numel(c);
    end

    if Km >= obj.max_contexts
        % All contexts instantiated: no novel slot beyond Km.
        weights(Km+1:end) = 0;
    elseif Km + 2 <= Cmax
        % Keep the Km used slots plus the single novel slot (Km+1); zero the rest.
        weights(Km+2:end) = 0;
    end

    weights(~isfinite(weights)) = 0;
    s = sum(weights);
    if s > 0
        weights = weights ./ s;
    end
end
