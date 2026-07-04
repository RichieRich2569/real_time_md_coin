function densities = retention_given_context_probability(obj, values)
%RETENTION_GIVEN_CONTEXT_PROBABILITY Per-context retention density on a grid.
%
%   densities = retention_given_context_probability(obj, values) returns a
%   containers.Map keyed by global context label, each value the posterior
%   density of that context's state-retention factor evaluated at the query
%   points values (a vector). Mirrors COIN's plot_retention_given_context and
%   the layout of state_given_context_probability, reading the aligned global
%   context moments (dynamics_mean(1,:) / dynamics_covar(1,1,:)).
%
%   Scalar model only (state_dim == 1): retention is a scalar-dynamics quantity
%   with no multi-dimensional counterpart (the MD model stores a dynamics matrix).
    arguments
        obj (1, 1) RealTimeCOIN
        values (:, :) double {mustBeFinite, mustBeReal}
    end
    mustBeScalarModel(obj, 'retention_given_context_probability');

    densities = containers.Map('KeyType', 'double', 'ValueType', 'any');
    alignment = ensureContextAlignment(obj);
    active = activeSummaryContexts(obj);
    active = active(active <= alignment.K);
    if isempty(active) && alignment.K > 0
        active = 1;
    end

    values = values(:)';
    M = alignment.global_contexts.dynamics_mean(1, :);
    V = squeeze(alignment.global_contexts.dynamics_covar(1, 1, :));
    for c = active
        densities(c) = obj.normal_pdf(values, M(c), V(c));
    end
end
