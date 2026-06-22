function P = stationaryStateCovMD(obj, A, Q)
%STATIONARYSTATECOVMD Stationary covariance of the MD latent AR(1) process.
%
%   For s_i = A s_{i-1} + d + w_i with w_i ~ N(0, Q) the stationary state
%   covariance solves the discrete Lyapunov equation
%       P = A P A' + Q.
%   Vectorising and using vec(AXB) = (B' (kron) A) vec(X) gives
%       vec(P) = (I - A (kron) A) \ vec(Q),
%   which we solve directly so no Control System Toolbox dependency
%   (dlyap) is required. This is the multi-dimensional generalisation of
%   stationaryStateVar.m (scalar Q/(1-a^2)). If the system is at/over the
%   stability boundary the Kronecker system is singular; we fall back to the
%   pseudo-inverse and symmetrise/clip so the result is a valid covariance.

    N = obj.state_dim;
    K = eye(N^2) - kron(A, A);
    qvec = Q(:);
    if rcond(K) < 1e-12
        pvec = pinv(K) * qvec;
    else
        pvec = K \ qvec;
    end
    P = reshape(pvec, N, N);
    P = (P + P') ./ 2;

    % Project to the nearest PSD matrix (clip negative eigenvalues) so a new
    % context is always seeded with a usable covariance.
    [Vc, Dc] = eig(P);
    dvals = max(real(diag(Dc)), 0);
    P = Vc * diag(dvals) * Vc';
    P = (P + P') ./ 2;
end
