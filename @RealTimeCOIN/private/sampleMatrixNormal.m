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
%
%   Inputs:
%     M  n-by-p mean matrix.
%     U  n-by-n among-row covariance.
%     V  p-by-p among-column covariance.
%
%   Output:
%     X  n-by-p matrix-normal sample.
%
%   See also sampleStableTheta, choljitter.

    if ~isnumeric(M) || ~isreal(M) || ~ismatrix(M)
        error("RealTimeCOIN:sampleMatrixNormal:invalidMean", ...
            "M must be a real numeric matrix.");
    end
    [n, p] = size(M);
    if ~isnumeric(U) || ~isreal(U) || ~isequal(size(U), [n, n])
        error("RealTimeCOIN:sampleMatrixNormal:invalidRowCov", ...
            "U must be a real %d-by-%d matrix matching size(M, 1).", n, n);
    end
    if ~isnumeric(V) || ~isreal(V) || ~isequal(size(V), [p, p])
        error("RealTimeCOIN:sampleMatrixNormal:invalidColCov", ...
            "V must be a real %d-by-%d matrix matching size(M, 2).", p, p);
    end

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
    % Escalating diagonal jitter: start one part in 1e12 of the matrix scale and
    % multiply by 10 each attempt (up to 8 tries, i.e. up to ~1e-4 * scale)
    % until chol succeeds. The 1e-12 base is negligible relative to V.
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
