function active = activeSummaryContexts(obj)
    weights = obj.contextProbabilityVector("predicted");
    active = find(weights > 0);
    if isempty(active)
        active = 1;
    end
end
