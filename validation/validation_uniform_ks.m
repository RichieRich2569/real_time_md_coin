function d = validation_uniform_ks(p)
%VALIDATION_UNIFORM_KS Kolmogorov-Smirnov distance from Uniform(0,1).
%
%   Probability integral transform values should be uniformly distributed
%   when the predictive distribution is calibrated.  This helper reports
%   the one-sample KS distance, i.e. the largest vertical gap between the
%   empirical CDF of the supplied PIT values and the CDF F(u)=u.

p = p(:);
p = p(isfinite(p));
p = sort(min(max(p, 0), 1));
n = numel(p);
if n == 0
    d = NaN;
    return;
end

upper = (1:n)' ./ n;
lower = (0:n-1)' ./ n;
d = max(max(abs(upper - p)), max(abs(lower - p)));
end
