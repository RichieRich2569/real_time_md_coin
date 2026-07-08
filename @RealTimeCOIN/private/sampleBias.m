function sampleBias(obj)
%SAMPLEBIAS Sample the per-context observation bias (scalar model).
%
%   sampleBias(obj) draws a scalar observation bias for every context and
%   particle from the conjugate Gaussian posterior implied by the
%   observation model y = s + b + v, v ~ N(0, obsVar), with the Gaussian
%   prior b ~ N(prior_mean_bias, prior_precision_bias^-1):
%       varB = 1 / (prior_precision_bias + bias_ss_2 / obsVar)
%       muB  = varB * (prior_precision_bias * prior_mean_bias + bias_ss_1 / obsVar).
%   Arrays are (max_contexts+1)-by-num_particles and updated in a single
%   vectorized pass. When infer_bias is false the bias is pinned to zero
%   (no sampling). See sampleBiasMD.m for the multivariate counterpart.

    if ~obj.infer_bias
        % Bias disabled: force b = 0 with zero posterior variance.
        obj.D.bias = zeros(obj.max_contexts+1, obj.num_particles);
        obj.D.bias_mean = obj.D.bias;
        obj.D.bias_var = zeros(size(obj.D.bias));
        return;
    end
    obsVar = obj.observationVariance();
    if obsVar == 0
        % Zero observation noise implies infinite data precision. Floor to eps
        % so bias_ss / obsVar stays finite and the posterior collapses onto the
        % empirical mean instead of yielding Inf/NaN (cf. qVar == 0 hack in
        % sampleDynamics.m).
        obsVar = eps;
    end
    varB = 1 ./ (obj.prior_precision_bias + obj.D.bias_ss_2 ./ obsVar);
    muB = varB .* (obj.prior_precision_bias .* obj.prior_mean_bias + obj.D.bias_ss_1 ./ obsVar);
    obj.D.bias_mean = muB;
    obj.D.bias_var = varB;
    % Untruncated Gaussian draw (bias is unconstrained on the real line).
    obj.D.bias = obj.sampleScalarNormal(muB, varB, size(muB), -Inf, Inf);
end
