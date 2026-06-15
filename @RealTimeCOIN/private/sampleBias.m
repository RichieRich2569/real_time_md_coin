function sampleBias(obj)
    if ~obj.infer_bias
        obj.D.bias = zeros(obj.max_contexts+1, obj.num_particles);
        obj.D.bias_mean = obj.D.bias;
        obj.D.bias_var = zeros(size(obj.D.bias));
        return;
    end
    obsVar = obj.observationVariance();
    if obsVar == 0
        obsVar = eps;
    end
    varB = 1 ./ (obj.prior_precision_bias + obj.D.bias_ss_2 ./ obsVar);
    muB = varB .* (obj.prior_precision_bias .* obj.prior_mean_bias + obj.D.bias_ss_1 ./ obsVar);
    obj.D.bias_mean = muB;
    obj.D.bias_var = varB;
    obj.D.bias = obj.sampleScalarNormal(muB, varB, size(muB), -Inf, Inf);
end
