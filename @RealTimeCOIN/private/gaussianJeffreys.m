function d = gaussianJeffreys(~, m1, v1, m2, v2)
%GAUSSIANJEFFREYS Jeffreys divergence between two scalar Gaussians.
%
%   d = gaussianJeffreys(obj, m1, v1, m2, v2) returns the (symmetric)
%   Jeffreys divergence between N(m1, v1) and N(m2, v2),
%       d = 0.5*(v1/v2 + v2/v1 + (m1-m2)^2*(1/v1 + 1/v2) - 2) >= 0,
%   which is the symmetrised KL divergence KL(1||2) + KL(2||1). It is used to
%   measure dissimilarity between per-context state beliefs. This is the scalar
%   counterpart of gaussianJeffreysMulti.m.

    % A non-finite (e.g. novel-context) variance is treated as maximally
    % diffuse: 1/eps is the largest finite precision we allow (eps is the
    % smallest resolvable spacing near 1, so 1/eps ~ 4.5e15).
    if ~isfinite(v1)
        v1 = 1 ./ eps;
    end
    if ~isfinite(v2)
        v2 = 1 ./ eps;
    end
    % Floor variances at eps to keep the reciprocals v1/v2, v2/v1 finite.
    v1 = max(v1, eps);
    v2 = max(v2, eps);
    d = 0.5 .* (v1 ./ v2 + v2 ./ v1 + (m1 - m2).^2 .* (1 ./ v1 + 1 ./ v2) - 2);
    if ~isfinite(d)
        d = realmax;  % sentinel: "infinitely divergent" but still finite
    end
    d = max(d, 0);    % divergence is non-negative; clip round-off below 0
end
