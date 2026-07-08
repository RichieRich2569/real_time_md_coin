function d = gaussianPdfColumnsMD(obj, X, m, S)
%GAUSSIANPDFCOLUMNSMD Multivariate Gaussian pdf evaluated at many query points.
%
%   d = gaussianPdfColumnsMD(obj, X, m, S) returns a 1-by-K row vector whose
%   k-th entry is the N-variate normal density N_N(X(:,k) | m, S), where X is
%   N-by-K (each column a query point), m is the N-by-1 mean and S the N-by-N
%   covariance. It is the vectorised, grid-evaluating counterpart of
%   gaussianLogLikChol.m: S is factored once by its lower Cholesky factor L
%   (S = L*L') and reused across all K points,
%
%       log N_N(x | m, S) = -0.5 * (N*log(2*pi) + log|S| + r' S^-1 r),
%       log|S| = 2 sum(log(diag(L))),   r' S^-1 r = (L\r)' (L\r),
%
%   so the per-point work is a triangular solve rather than a fresh O(N^3)
%   factorisation. At N == 1 this reduces to RealTimeCOIN.normal_pdf.

    N = size(X, 1);
    K = size(X, 2);
    [L, ~] = obj.choljitter(S);            % diagonal fallback already PSD-safe
    logDetS = 2 * sum(log(diag(L)));
    R = X - m(:);                          % N-by-K residuals (implicit expand)
    foo = L \ R;                           % N-by-K
    mahalanobis = sum(foo.^2, 1);          % 1-by-K
    logPdf = -0.5 * (N * log(2*pi) + logDetS + mahalanobis);
    d = exp(logPdf);
    d(~isfinite(d)) = realmax;   % sentinel: largest finite double for overflow
    d = reshape(d, 1, K);
end
