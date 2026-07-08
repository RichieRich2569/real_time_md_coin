function Theta = sampleStableTheta(obj, M, U, V)
%SAMPLESTABLETHETA Draw a stable augmented dynamics matrix Theta = [A | d].
%
%   Theta = sampleStableTheta(obj, M, U, V) draws Theta ~ MN(M, U, V) and
%   enforces a bounded-stability constraint on the dynamics block
%   A = Theta(:, 1:N): the spectral radius rho(A) = max(abs(eig(A))) must be
%   < 1 so the latent AR(1) process is stationary. This is the
%   multi-dimensional analogue of the scalar a in [0, 1) truncation in
%   sampleBivariateTruncated.m. It also implies abs(det(A)) < 1 because the
%   determinant magnitude is the product of the eigenvalue magnitudes.
%
%   Strategy (matching the execution plan): rejection-sample up to max_iter
%   times; if no stable draw is found, force stability by spectral scaling,
%   A <- A / (rho + margin), which shrinks every eigenvalue inside the unit
%   disc while leaving the drift column d unchanged.
%
%   Inputs:
%     M  N-by-(N+1) mean of the augmented dynamics [A | d].
%     U  N-by-N among-row covariance.
%     V  (N+1)-by-(N+1) among-column covariance.
%
%   Output:
%     Theta  N-by-(N+1) stable augmented dynamics with spectral radius < 1.
%
%   See also sampleMatrixNormal, sampleBivariateTruncated.

    N = obj.state_dim;
    if ~isnumeric(M) || ~isreal(M) || ~isequal(size(M), [N, N + 1])
        error("RealTimeCOIN:sampleStableTheta:invalidMean", ...
            "M must be a real %d-by-%d matrix ([A | d]).", N, N + 1);
    end

    % maxIter: number of rejection-sampling attempts before falling back to
    % spectral scaling. margin: the safety gap kept below the unit circle when
    % rescaling, A <- A / (rho + margin), so the projected spectral radius is
    % rho / (rho + margin) < 1 strictly.
    maxIter = 10;
    margin = 0.01;

    Theta = M;          % Sensible default if sampling repeatedly fails.
    A = Theta(:, 1:N);
    rho = Inf;
    for iter = 1:maxIter
        Theta = obj.sampleMatrixNormal(M, U, V);
        A = Theta(:, 1:N);
        rho = max(abs(eig(A)));
        if isfinite(rho) && rho < 1
            return;
        end
    end

    if ~isfinite(rho) || rho <= 0
        % Degenerate draw: fall back to the prior/posterior mean, then apply
        % the same stability projection below if needed.
        Theta = M;
        A = Theta(:, 1:N);
        rho = max(abs(eig(A)));
    end

    if ~isfinite(rho) || rho <= 0
        Theta(:, 1:N) = zeros(N);
        return;
    end

    if rho < 1
        return;
    end
    Theta(:, 1:N) = A ./ (rho + margin);
end
