function Theta = sampleStableTheta(obj, M, U, V)
%SAMPLESTABLETHETA Draw a stable augmented dynamics matrix Theta = [A | d].
%
%   Theta = sampleStableTheta(obj, M, U, V) draws Theta ~ MN(M, U, V) and
%   enforces a bounded-stability constraint on the dynamics block
%   A = Theta(:, 1:N): the spectral radius rho(A) = max(abs(eig(A))) must be
%   < 1 so the latent AR(1) process is stationary. This is the
%   multi-dimensional analogue of the scalar a in [0, 1) truncation in
%   sampleBivariateTruncated.m.
%
%   Strategy (matching the execution plan): rejection-sample up to max_iter
%   times; if no stable draw is found, force stability by spectral scaling,
%   A <- A / (rho + margin), which shrinks every eigenvalue inside the unit
%   disc while leaving the drift column d unchanged.

    N = obj.state_dim;
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
        % Degenerate draw: fall back to the (stable) prior/posterior mean.
        Theta = M;
        return;
    end
    Theta(:, 1:N) = A ./ (rho + margin);
end
