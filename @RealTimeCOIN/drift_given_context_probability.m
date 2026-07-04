function densities = drift_given_context_probability(obj, values)
%DRIFT_GIVEN_CONTEXT_PROBABILITY Per-context drift density on a grid.
%
%   densities = drift_given_context_probability(obj, values) returns a
%   containers.Map keyed by global context label, each value the posterior
%   density of that context's state drift evaluated at the query points values
%   (a vector). Mirrors COIN's plot_drift_given_context, reading the aligned
%   global context moments (dynamics_mean(2,:) / dynamics_covar(2,2,:)).
%
%   Scalar model only (state_dim == 1); see retention_given_context_probability.
    arguments
        obj (1, 1) RealTimeCOIN
        values (:, :) double {mustBeFinite, mustBeReal}
    end
    mustBeScalarModel(obj, 'drift_given_context_probability');

    densities = containers.Map('KeyType', 'double', 'ValueType', 'any');
    alignment = ensureContextAlignment(obj);
    active = activeSummaryContexts(obj);
    active = active(active <= alignment.K);
    if isempty(active) && alignment.K > 0
        active = 1;
    end

    values = values(:)';
    M = alignment.global_contexts.dynamics_mean(2, :);
    V = squeeze(alignment.global_contexts.dynamics_covar(2, 2, :));
    for c = active
        densities(c) = obj.normal_pdf(values, M(c), V(c));
    end
end
