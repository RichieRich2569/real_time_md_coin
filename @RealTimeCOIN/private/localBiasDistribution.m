function [mu, variance] = localBiasDistribution(obj, local, p)
%LOCALBIASDISTRIBUTION Gaussian observation-bias belief for one context/particle.
%   [mu, variance] = localBiasDistribution(obj, local, p) returns the mean and
%   variance of the (scalar) observation-bias posterior for local context slot
%   `local` of particle `p`.
%
%   The posterior sufficient statistics (bias_mean/bias_var) are used when they
%   have been computed for this slot; otherwise the routine falls back to the
%   prior: the stored bias sample with the prior bias precision converted to a
%   variance. When bias inference is disabled the bias is deterministically 0.
%
%   Inputs:
%     local  local context slot index.
%     p      particle index.
%
%   Outputs:
%     mu        posterior bias mean (0 when infer_bias is false).
%     variance  posterior bias variance, clamped to be finite and non-negative.
    if obj.infer_bias && isfield(obj.D, 'bias_mean') && ...
            size(obj.D.bias_mean, 1) >= local && size(obj.D.bias_mean, 2) >= p
        mu = obj.D.bias_mean(local, p);
        variance = obj.D.bias_var(local, p);
    elseif obj.infer_bias
        mu = obj.D.bias(local, p);
        variance = obj.precisionToVariance(obj.prior_precision_bias);
    else
        mu = 0;
        variance = 0;
    end
    if ~isfinite(variance)
        variance = 1 ./ eps;    % non-finite precision -> effectively unbounded variance
    end
    variance = max(variance, 0);
end
