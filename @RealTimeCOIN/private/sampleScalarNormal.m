function X = sampleScalarNormal(obj, mu, variance, sz, low, high)
    if isscalar(mu)
        mu = mu .* ones(sz);
    end
    if isscalar(variance)
        variance = variance .* ones(sz);
    end
    if isscalar(low)
        low = low .* ones(sz);
    end
    if isscalar(high)
        high = high .* ones(sz);
    end

    X = mu;
    stochastic = variance > 0 & isfinite(variance);
    if any(stochastic, 'all')
        sigma = sqrt(variance(stochastic));
        l = (low(stochastic) - mu(stochastic)) ./ sigma;
        u = (high(stochastic) - mu(stochastic)) ./ sigma;
        X(stochastic) = mu(stochastic) + sigma .* obj.trandn(l, u);
    end
    X = min(max(X, low), high - eps);
end
