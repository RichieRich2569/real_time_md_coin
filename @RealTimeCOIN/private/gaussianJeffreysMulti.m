function d = gaussianJeffreysMulti(obj, m1, s1, m2, s2)
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
        d = realmax;
    end
    d = max(d, 0);
end
