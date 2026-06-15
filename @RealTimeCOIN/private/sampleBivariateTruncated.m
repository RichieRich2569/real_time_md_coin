function x = sampleBivariateTruncated(~, mu, covar)
    if any(~isfinite(covar), 'all') || any(~isfinite(mu))
        x = [min(max(mu(1), 0), 1-eps); mu(2)];
        return;
    end
    covar = (covar + covar') ./ 2 + 1e-12 .* eye(2);
    [L, flag] = chol(covar, 'lower');
    if flag ~= 0
        x = [min(max(mu(1), 0), 1-eps); mu(2)];
        return;
    end
    for attempt = 1:50
        candidate = mu + L * randn(2,1);
        if candidate(1) >= 0 && candidate(1) < 1
            x = candidate;
            return;
        end
    end
    x = [min(max(mu(1), 0), 1-eps); mu(2)];
end
