function [mu, v] = poolMoments(mus, vs, N)
%POOLMOMENTS Moments of the equal-weight pooled mixture over runs.
%   [mu, v] = poolMoments(mus, vs, N) combines per-run predictive-state moments
%   into the moments of the pooled mixture that gives each run weight 1/R (the
%   law of total (co)variance), NOT a naive mean of the per-run covariances:
%       mu = (1/R) sum_k mu_k
%       v  = (1/R) sum_k ( v_k + mu_k mu_k' ) - mu mu'.
%
%   mus is a 1-by-R cell of N-by-1 mean vectors (scalars when N == 1); vs is a
%   1-by-R cell of N-by-N covariances (scalars when N == 1). The combination is
%   NaN-aware at run granularity: a run contributes only if its mean and
%   covariance are entirely finite; if no run is valid the result is NaN. For
%   N == 1 the variance is floored at 0; for N > 1 it is symmetrised.
    R = numel(mus);
    valid = false(1, R);
    for k = 1:R
        valid(k) = all(isfinite(mus{k}(:))) && all(isfinite(vs{k}(:)));
    end
    idx = find(valid);

    if isempty(idx)
        mu = nan(N, 1);
        if N == 1
            v = NaN;
        else
            v = nan(N, N);
        end
        return;
    end

    mu = zeros(N, 1);
    second = zeros(N, N);
    for k = idx
        mk = mus{k}(:);
        mu = mu + mk;
        second = second + (vs{k} + mk * mk');
    end
    m = numel(idx);
    mu = mu ./ m;
    second = second ./ m;
    v = second - (mu * mu');

    if N == 1
        v = max(v, 0);
    else
        v = (v + v') ./ 2;
    end
end
