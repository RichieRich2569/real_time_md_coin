function d = gaussianJeffreysMulti(obj, m1, s1, m2, s2)
%GAUSSIANJEFFREYSMULTI Jeffreys divergence between two multivariate Gaussians.
%
%   d = gaussianJeffreysMulti(obj, m1, s1, m2, s2) returns the symmetric
%   Jeffreys divergence between N(m1, s1) and N(m2, s2) with k-dimensional
%   means m1, m2 and covariances s1, s2,
%       d = 0.5*(trace(s2^-1 s1 + s1^-1 s2) + delta'(s1^-1 + s2^-1)delta - 2k),
%   where delta = m1 - m2. It is the multi-dimensional generalisation of
%   gaussianJeffreys.m and collapses to it at k == 1. Covariances are
%   regularised (regularizeCovariance) and inverted stably (safeInverse) before
%   use so slightly non-PD particle covariances do not throw.

    m1 = m1(:);
    m2 = m2(:);
    s1 = obj.regularizeCovariance(s1);
    s2 = obj.regularizeCovariance(s2);
    inv1 = obj.safeInverse(s1);
    inv2 = obj.safeInverse(s2);
    delta = m1 - m2;
    k = numel(m1);
    d = 0.5 .* (trace(inv2 * s1 + inv1 * s2) + delta' * (inv1 + inv2) * delta - 2 .* k);
    if ~isfinite(d)
        d = realmax;  % sentinel: "infinitely divergent" but still finite
    end
    d = max(d, 0);    % divergence is non-negative; clip round-off below 0
end
