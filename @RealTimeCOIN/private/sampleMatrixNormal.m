function X = sampleMatrixNormal(obj, M, U, V)
%SAMPLEMATRIXNORMAL Draw X ~ MN_{n,p}(M, U, V).
%
%   X = sampleMatrixNormal(obj, M, U, V) returns an n-by-p sample from the
%   matrix-normal distribution with mean M (n-by-p), among-row covariance U
%   (n-by-n) and among-column covariance V (p-by-p), i.e.
%   vec(X) ~ N(vec(M), V (kron) U).
%
%   Sampling identity: if Z is n-by-p standard normal and L_U L_U' = U,
%   R_V' R_V = V, then
%       X = M + L_U * Z * R_V
%   has the required first two moments, because
%       Cov(vec(X)) = (R_V' R_V) (kron) (L_U L_U') = V (kron) U.
%   We take L_U as the lower Cholesky factor of U and R_V as the upper
%   Cholesky factor of V. choljitter is used for U so that a (near-)singular
%   row covariance (e.g. zero process noise) degrades gracefully.

    [n, p] = size(M);
    Z = randn(n, p);

    [Lu, ~] = obj.choljitter(U);                 % Lu * Lu' = U  (lower)
    Rv = cholUpperPSD((V + V') ./ 2);            % Rv' * Rv = V  (upper)

    X = M + Lu * Z * Rv;
end

function R = cholUpperPSD(V)
%CHOLUPPERPSD Upper Cholesky factor of a symmetric PSD matrix with jitter.
    [R, flag] = chol(V);                         % V = R' * R (upper)
    if flag == 0
        return;
    end
    scale = mean(diag(V));
    if ~isfinite(scale) || scale <= 0
        scale = 1;
    end
    jit = 1e-12 * scale;
    for k = 1:8
        [R, flag] = chol(V + jit * eye(size(V)));
        if flag == 0
            return;
        end
        jit = jit * 10;
    end
    R = diag(sqrt(max(diag(V), eps)));
end
