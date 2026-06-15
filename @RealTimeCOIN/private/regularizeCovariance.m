function covar = regularizeCovariance(~, covar)
    covar(~isfinite(covar)) = 0;
    covar = (covar + covar') ./ 2;
    if isempty(covar)
        covar = eps;
        return;
    end
    covar = covar + eps .* eye(size(covar));
    if rcond(covar) < 1e-12
        covar = covar + 1e-9 .* eye(size(covar));
    end
end
