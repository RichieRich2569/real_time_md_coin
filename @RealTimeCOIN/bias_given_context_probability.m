function densities = bias_given_context_probability(obj, values)
%BIAS_GIVEN_CONTEXT_PROBABILITY Per-context measurement-bias density on a grid.
%
%   densities = bias_given_context_probability(obj, values) returns a
%   containers.Map keyed by global context label, each value the posterior
%   density of that context's measurement bias evaluated at the query points
%   values (a vector). Mirrors COIN's plot_bias_given_context, reading the
%   aligned global context moments (bias_mean / bias_var).
%
%   Requires infer_bias == true (as in COIN). Scalar model only (state_dim == 1):
%   the multi-dimensional prototypes track a bias mean but no bias covariance.
    arguments
        obj (1, 1) RealTimeCOIN
        values (:, :) double {mustBeFinite, mustBeReal}
    end
    mustBeScalarModel(obj, 'bias_given_context_probability');
    if ~obj.infer_bias
        error('RealTimeCOIN:BiasNotInferred', ...
            'bias_given_context_probability requires infer_bias == true.');
    end

    densities = containers.Map('KeyType', 'double', 'ValueType', 'any');
    alignment = ensureContextAlignment(obj);
    active = clampActiveSummaryContexts(obj, alignment);

    values = values(:)';
    M = alignment.global_contexts.bias_mean;
    V = alignment.global_contexts.bias_var;
    for c = active
        densities(c) = obj.normal_pdf(values, M(c), V(c));
    end
end
