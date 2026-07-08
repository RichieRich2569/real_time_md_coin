function covar = regularizeCovariance(~, covar)
%REGULARIZECOVARIANCE Symmetrise and condition a covariance matrix.
%
%   covar = regularizeCovariance(obj, covar) sanitises a possibly ill-formed
%   particle covariance so downstream inversion/factorisation is safe: it zeroes
%   non-finite entries, symmetrises, and adds diagonal loading to guarantee
%   positive definiteness and a workable condition number. Used before
%   safeInverse in the Jeffreys/Kalman MD paths.

    covar(~isfinite(covar)) = 0;
    covar = (covar + covar') ./ 2;   % enforce exact symmetry
    if isempty(covar)
        covar = eps;                 % empty -> smallest positive scalar variance
        return;
    end
    % Add eps on the diagonal (eps = smallest resolvable spacing near 1) so a
    % zero/rank-deficient covariance becomes strictly positive definite.
    covar = covar + eps .* eye(size(covar));
    % rcond < 1e-12 flags a near-singular matrix (reciprocal condition number
    % threshold); bump the diagonal by 1e-9 to restore a usable condition.
    if rcond(covar) < 1e-12
        covar = covar + 1e-9 .* eye(size(covar));
    end
end
