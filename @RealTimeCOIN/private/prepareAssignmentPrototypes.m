function prepared = prepareAssignmentPrototypes(obj, Km, prototypes)
%PREPAREASSIGNMENTPROTOTYPES Precompute per-prototype terms for the MD cost.
%   prepared = prepareAssignmentPrototypes(obj, Km, prototypes) caches the
%   prototype quantities that assignmentCostMatrixMD reuses for every modal
%   particle within a sweep, so they are computed once per sweep rather than once
%   per (particle, context) pair:
%     state_cov  - regularised N-by-N-by-Km prototype state covariances.
%     state_inv  - their inverses (for the Gaussian-Jeffreys divergence).
%     theta_vec  - vectorised augmented dynamics [A | d], N*(N+1)-by-Km.
%   For the scalar model (state_dim <= 1) there is nothing to precompute and an
%   empty struct is returned.

    prepared = struct();
    if obj.state_dim <= 1
        return;   % scalar cost path does not use precomputed prototype terms
    end

    N = obj.state_dim;
    prepared.state_cov = zeros(N, N, Km);
    prepared.state_inv = zeros(N, N, Km);
    prepared.theta_vec = zeros(N * (N + 1), Km);
    for globalIdx = 1:Km
        covar = obj.regularizeCovariance(prototypes.state_cov(:, :, globalIdx));
        prepared.state_cov(:, :, globalIdx) = covar;
        prepared.state_inv(:, :, globalIdx) = obj.safeInverse(covar);
        prepared.theta_vec(:, globalIdx) = reshape(prototypes.theta_mean(:, :, globalIdx), [], 1);
    end
end
