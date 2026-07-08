function x = sampleBivariateTruncated(obj, mu, covar)
%SAMPLEBIVARIATETRUNCATED Draw [a; d] from a bivariate normal, a truncated.
%   x = sampleBivariateTruncated(obj, mu, covar) returns a 2-by-1 sample from
%   the bivariate normal N(mu, covar) in which the FIRST component (the
%   retention a) is truncated to the interval [0, 1) while the second component
%   (the drift d) is left unconstrained. This is the conjugate posterior draw of
%   the scalar dynamics [a; d]; the [0, 1) constraint keeps the latent AR(1)
%   retention in its stationary range (the multi-dimensional analogue is the
%   spectral-radius constraint in sampleStableTheta).
%
%   Method. The lower Cholesky factor L of covar is used to draw a standardised
%   z ~ N(0, I) with its first coordinate truncated to the standardised interval
%   [(0 - mu(1))/L(1,1), (1 - mu(1))/L(1,1)] (via trandn), then x = mu + L*z.
%   Because L is lower triangular, x(1) = mu(1) + L(1,1)*z(1) depends only on the
%   truncated first coordinate, so truncating z(1) truncates a exactly.
%
%   Degenerate / numerical fallbacks:
%     * Non-finite mu or covar, or a covar that is not positive definite (chol
%       fails or L(1,1) <= 0): return the deterministic mean with a clamped to
%       [0, 1 - eps].
%     * covar is symmetrised and given a 1e-12 diagonal jitter before the
%       Cholesky. The 1e-12 is a small ridge that regularises a (near-)singular
%       posterior covariance so chol succeeds; it is negligible relative to the
%       parameter scales and does not bias the draw materially.
%     * The final x(1) is clamped to [0, 1 - eps] so the returned retention is
%       strictly below 1 even after rounding.
%
%   Inputs:
%     mu     2-by-1 (or 1-by-2) posterior mean [a; d].
%     covar  2-by-2 posterior covariance.
%
%   Output:
%     x      2-by-1 sample [a; d] with a in [0, 1).
%
%   See also trandn, sampleScalarNormal, sampleStableTheta.

    if ~isnumeric(mu) || ~isreal(mu) || numel(mu) ~= 2
        error("RealTimeCOIN:sampleBivariateTruncated:invalidMean", ...
            "mu must be a real 2-element vector.");
    end
    if ~isnumeric(covar) || ~isreal(covar) || ~isequal(size(covar), [2, 2])
        error("RealTimeCOIN:sampleBivariateTruncated:invalidCovariance", ...
            "covar must be a real 2-by-2 matrix.");
    end

    if any(~isfinite(covar), 'all') || any(~isfinite(mu))
        x = [min(max(mu(1), 0), 1 - eps); mu(2)];
        return;
    end
    % Symmetrise and add a tiny 1e-12 ridge so a (near-)singular posterior
    % covariance still yields a valid Cholesky factor.
    covar = (covar + covar') ./ 2 + 1e-12 .* eye(2);
    [L, flag] = chol(covar, 'lower');
    if flag ~= 0 || L(1, 1) <= 0
        x = [min(max(mu(1), 0), 1 - eps); mu(2)];
        return;
    end

    l = [(0 - mu(1)) ./ L(1, 1); -Inf];
    u = [(1 - mu(1)) ./ L(1, 1); Inf];
    z = obj.trandn(l, u);
    x = mu(:) + L * z;
    x(1) = min(max(x(1), 0), 1 - eps);
end
