function p = validation_mixture_cdf(x, weights, means, variances)
%VALIDATION_MIXTURE_CDF CDF of a Gaussian mixture at x.

w = weights;
w(~isfinite(w) | w < 0) = 0;
total = sum(w, 'all');
if total <= 0
    p = NaN;
    return;
end
w = w ./ total;
p = sum(w .* RealTimeCOIN.normal_cdf(x, means, max(variances, 0)), 'all');
p = min(max(p, 0), 1);
end
