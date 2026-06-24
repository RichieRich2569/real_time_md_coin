function prepared = prepareAssignmentPrototypes(obj, Km, prototypes)
    prepared = struct();
    if obj.state_dim <= 1
        return;
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
