function densities = state_given_context_probability(obj, values)
    values = values(:)';
    densities = containers.Map('KeyType', 'double', 'ValueType', 'any');
    alignment = ensureContextAlignment(obj);
    M = alignment.global_contexts.state_mean;
    V = alignment.global_contexts.state_var;
    active = activeSummaryContexts(obj);
    active = active(active <= alignment.K);
    if isempty(active) && alignment.K > 0
        active = 1;
    end
    for c = active
        d = obj.normal_pdf(values, M(c), V(c));
        densities(c) = d;
    end
end
