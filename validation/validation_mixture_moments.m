function [mu, v] = validation_mixture_moments(weights, means, variances)
%VALIDATION_MIXTURE_MOMENTS Mean and variance of a Gaussian mixture.
%
%   The RealTimeCOIN predictive distributions are represented as mixtures
%   across contexts and particles.  If component k has weight w_k, mean
%   m_k and variance V_k, then
%
%       E[X] = sum_k w_k m_k,
%       Var[X] = sum_k w_k (V_k + m_k^2) - E[X]^2.

w = weights;
w(~isfinite(w) | w < 0) = 0;
total = sum(w, 'all');
if total <= 0
    mu = NaN;
    v = NaN;
    return;
end
w = w ./ total;
variances = max(variances, 0);
mu = sum(w .* means, 'all');
second = sum(w .* (variances + means.^2), 'all');
v = max(second - mu.^2, 0);
end
