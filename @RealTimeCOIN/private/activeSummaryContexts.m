function active = activeSummaryContexts(obj)
%ACTIVESUMMARYCONTEXTS Global context slots carrying non-zero probability.
%   active = activeSummaryContexts(obj) returns the indices of the global
%   context slots whose predicted probability is greater than zero, i.e. the
%   contexts worth reporting in a summary. When no slot carries mass it falls
%   back to slot 1 so callers always receive at least one context.
%
%   Output:
%     active  row vector of active global context slot indices (never empty).
    weights = obj.contextProbabilityVector("predicted");
    active = find(weights > 0);
    if isempty(active)
        active = 1;
    end
end
