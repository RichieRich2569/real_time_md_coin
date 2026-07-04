function densities = bias_probability(obj, values)
%BIAS_PROBABILITY Marginal (across-context) measurement-bias density on a grid.
%
%   densities = bias_probability(obj, values) returns the posterior bias density
%   marginalised over contexts and particles, using the predicted context
%   probabilities as mixing weights,
%       p(b) = (1/P) sum_p sum_c W(c,p) N(b | bias_mean_{c,p}, bias_var_{c,p}),
%   mirroring COIN's plot_bias / compute_marginal_distribution. values is a
%   vector of query points and densities a row vector of equal length.
%
%   Requires infer_bias == true. Scalar model only (state_dim == 1).
    arguments
        obj (1, 1) RealTimeCOIN
        values (:, :) double {mustBeFinite, mustBeReal}
    end
    mustBeScalarModel(obj, 'bias_probability');
    if ~obj.infer_bias
        error('RealTimeCOIN:BiasNotInferred', ...
            'bias_probability requires infer_bias == true.');
    end

    values = values(:)';
    densities = zeros(size(values));
    W = obj.D.predicted_probabilities;
    P = obj.num_particles;
    Cmax = obj.max_contexts + 1;
    for p = 1:P
        for c = 1:Cmax
            if W(c, p) > 0
                [m, v] = obj.localBiasDistribution(c, p);
                densities = densities + W(c, p) .* obj.normal_pdf(values, m, v);
            end
        end
    end
    densities = densities ./ P;
end
