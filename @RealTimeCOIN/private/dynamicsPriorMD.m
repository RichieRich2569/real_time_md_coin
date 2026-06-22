function [M0, V0inv, V0] = dynamicsPriorMD(obj)
%DYNAMICSPRIORMD Matrix-normal prior on the augmented dynamics Theta = [A | d].
%
%   Returns the prior mean M0 (N-by-(N+1)) and the prior column-covariance
%   precision V0inv and covariance V0 ((N+1)-by-(N+1)) for the matrix-normal
%   prior Theta ~ MN(M0, U = Q, V0), where Q is the process-noise (row)
%   covariance. The reduction-to-scalar mapping is the key design point:
%
%     M0    = [prior_mean_retention * I_N , prior_mean_drift * 1_N]
%     V0inv = sigma_process_noise^2 * diag([prec_ret*ones(1,N), prec_drift])
%
%   With U = Q the posterior column covariance and mean
%       V_post = (V0inv + Lambda_xx)^-1
%       M_post = (M0 * V0inv + Lambda_yx) * V_post
%   then reduce ALGEBRAICALLY to the scalar update in sampleDynamics.m at
%   N == 1: there Cov([a;d]) = U * V_post = (priorPrec + Lambda_xx/sigma^2)^-1,
%   exactly the scalar covar = inv(priorPrec + ss2/qVar). See sampleDynamicsMD.m
%   for the full derivation comment.
%
%   sigma_process_noise^2 is used as the reference scale even when a custom
%   process_noise_covariance is supplied, so that the isotropic default
%   reproduces the scalar prior; for a custom Q the per-entry prior variance
%   of A then scales with diag(Q) (documented behaviour). A zero scale is
%   floored to eps, mirroring the qVar == 0 guard in sampleDynamics.m.

    N = obj.state_dim;

    M0 = [obj.prior_mean_retention * eye(N), obj.prior_mean_drift * ones(N, 1)];

    s2 = obj.sigma_process_noise^2;
    if s2 == 0
        s2 = eps;
    end
    precDiag = [obj.prior_precision_retention * ones(1, N), obj.prior_precision_drift];
    V0inv = s2 * diag(precDiag);

    % V0 is diagonal with strictly positive entries, so the inverse is exact
    % and well conditioned.
    V0 = diag(1 ./ diag(V0inv));
end
