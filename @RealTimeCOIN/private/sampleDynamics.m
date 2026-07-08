function sampleDynamics(obj)
%SAMPLEDYNAMICS Sample per-context linear dynamics [a; d] (scalar model).
%
%   sampleDynamics(obj) draws the retention/drift pair [a; d] for every
%   context c and particle p from the conjugate bivariate-normal posterior
%   implied by the state-space regression s_i = a * s_{i-1} + d + w_i with
%   process noise w_i ~ N(0, qVar). Combining the Gaussian prior
%   N(priorMean, priorPrec^-1) with the accumulated sufficient statistics
%   (Gram matrix ss2 and cross term ss1) gives
%       covar = (priorPrec + ss2 / qVar)^-1
%       mu    = covar * (priorPrec * priorMean + ss1 / qVar).
%   The draw is truncated to the stable, causal region a in [0, 1) by
%   sampleBivariateTruncated. This is the scalar special case (N == 1) of
%   sampleDynamicsMD.m; see that file for the full matrix-normal derivation.

    Cmax = obj.max_contexts + 1;                % context slots incl. novel context
    P = obj.num_particles;
    priorPrec = diag([obj.prior_precision_retention, obj.prior_precision_drift]);
    priorMean = [obj.prior_mean_retention; obj.prior_mean_drift];
    qVar = obj.sigma_process_noise^2;           % process-noise variance (scales the data precision)
    for p = 1:P
        for c = 1:Cmax
            ss2 = squeeze(obj.D.dynamics_ss_2(c,p,:,:));  % 2x2 regressor Gram matrix
            ss1 = squeeze(obj.D.dynamics_ss_1(c,p,:));    % 2x1 regressor-response cross term
            if qVar == 0
                % Zero process noise means a deterministic AR(1): the data
                % carry infinite precision. Divide by eps instead of 0 so the
                % posterior collapses (near-)exactly onto the least-squares fit
                % without producing Inf/NaN. Mirrors the s2 == 0 guard in
                % dynamicsPriorMD.m and the obsVar == 0 guard in sampleBias.m.
                covar = obj.safeInverse(priorPrec + ss2 ./ eps);
                mu = covar * (priorPrec * priorMean + ss1 ./ eps);
            else
                covar = obj.safeInverse(priorPrec + ss2 ./ qVar);
                mu = covar * (priorPrec * priorMean + ss1 ./ qVar);
            end
            % Draw [a; d] truncated to the stable retention region a in [0, 1).
            sample = obj.sampleBivariateTruncated(mu, covar);
            obj.D.retention(c,p) = sample(1);
            obj.D.drift(c,p) = sample(2);
            obj.D.dynamics_mean(:,c,p) = mu;
            obj.D.dynamics_covar(:,:,c,p) = covar;
        end
    end
end
