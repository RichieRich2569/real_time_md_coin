function X = sampleScalarNormal(~, mu, variance, sz, low, high)
    if isscalar(mu)
        mu = mu .* ones(sz);
    end
    if isscalar(variance)
        variance = variance .* ones(sz);
    end
    X = mu;
    stochastic = variance > 0 & isfinite(variance);
    X(stochastic) = mu(stochastic) + sqrt(variance(stochastic)) .* randn(sum(stochastic,'all'),1);
    X = min(max(X, low), high - eps);
end
