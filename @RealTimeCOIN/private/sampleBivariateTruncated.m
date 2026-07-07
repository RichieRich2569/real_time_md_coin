function x = sampleBivariateTruncated(obj, mu, covar)
    if any(~isfinite(covar), 'all') || any(~isfinite(mu))
        x = [min(max(mu(1), 0), 1-eps); mu(2)];
        return;
    end
    covar = (covar + covar') ./ 2 + 1e-12 .* eye(2);
    [L, flag] = chol(covar, 'lower');
    if flag ~= 0 || L(1,1) <= 0
        x = [min(max(mu(1), 0), 1-eps); mu(2)];
        return;
    end

    l = [(0 - mu(1)) ./ L(1,1); -Inf];
    u = [(1 - mu(1)) ./ L(1,1); Inf];
    z = obj.trandn(l, u);
    x = mu(:) + L * z;
    x(1) = min(max(x(1), 0), 1 - eps);
end
