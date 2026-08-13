function weights = contextProbabilityVectorCore(kind, predictedFcn, responsibilitiesFcn, countFcn, edges)
%CONTEXTPROBABILITYVECTORCORE Shared kind-dispatch for context probability vectors.
%   weights = contextProbabilityVectorCore(kind, predictedFcn, ...
%       responsibilitiesFcn, countFcn, edges) computes the raw (unnormalised)
%   context probability vector selected by kind. It is the common core shared by
%   contextProbabilityVector (global frame) and localContextProbabilityVector
%   (local frame); each caller supplies frame-specific inputs and applies its own
%   trailing-slot handling and final renormalisation.
%
%   Only the source for the selected kind is evaluated, so the frame-specific
%   sources are passed as zero-argument function handles:
%     predictedFcn        returns the per-particle predicted-probability matrix.
%     responsibilitiesFcn returns the per-particle responsibilities matrix.
%     countFcn            returns the row vector of sampled context labels.
%   edges is the integer-centred histcounts edge vector used by the "count" kind.
%
%   kind selects the quantity summarised:
%     "predicted"        mean of the predicted context probabilities.
%     "responsibilities" mean of the context responsibilities.
%     "count"            empirical frequency of the sampled contexts.
%
%   Output:
%     weights  1-by-numel(edges)-1 row of unnormalised weights. Callers are
%              responsible for any slot zeroing and the guarded renormalisation.
    switch kind
        case "predicted"
            W = predictedFcn();
            weights = mean(W, 2)';
        case "responsibilities"
            W = responsibilitiesFcn();
            weights = mean(W, 2)';
        case "count"
            c = countFcn();
            weights = histcounts(c, edges) ./ max(numel(c), 1);
    end
end
