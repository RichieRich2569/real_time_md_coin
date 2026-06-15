function [mu, variance] = localBiasDistribution(obj, local, p)
    if obj.infer_bias && isfield(obj.D, 'bias_mean') && ...
            size(obj.D.bias_mean, 1) >= local && size(obj.D.bias_mean, 2) >= p
        mu = obj.D.bias_mean(local,p);
        variance = obj.D.bias_var(local,p);
    elseif obj.infer_bias
        mu = obj.D.bias(local,p);
        variance = obj.precisionToVariance(obj.prior_precision_bias);
    else
        mu = 0;
        variance = 0;
    end
    if ~isfinite(variance)
        variance = 1 ./ eps;
    end
    variance = max(variance, 0);
end
