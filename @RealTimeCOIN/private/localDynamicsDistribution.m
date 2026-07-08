function [mu, covar] = localDynamicsDistribution(obj, local, p)
%LOCALDYNAMICSDISTRIBUTION Gaussian dynamics belief for one context/particle.
%   [mu, covar] = localDynamicsDistribution(obj, local, p) returns the mean and
%   covariance of the joint [retention; drift] dynamics posterior for local
%   context slot `local` of particle `p` (scalar-state model).
%
%   The posterior sufficient statistics (dynamics_mean/dynamics_covar) are used
%   when available; otherwise the routine falls back to the prior: the stored
%   [retention; drift] samples with a diagonal covariance built from the prior
%   retention and drift precisions. The covariance is regularised before return.
%
%   Inputs:
%     local  local context slot index.
%     p      particle index.
%
%   Outputs:
%     mu     2-by-1 mean [retention; drift].
%     covar  2-by-2 regularised covariance.
    if isfield(obj.D, 'dynamics_mean') && ...
            size(obj.D.dynamics_mean, 2) >= local && size(obj.D.dynamics_mean, 3) >= p
        mu = obj.D.dynamics_mean(:, local, p);
    else
        mu = [obj.D.retention(local, p); obj.D.drift(local, p)];
    end

    if isfield(obj.D, 'dynamics_covar') && ...
            size(obj.D.dynamics_covar, 3) >= local && size(obj.D.dynamics_covar, 4) >= p
        covar = obj.D.dynamics_covar(:, :, local, p);
    else
        covar = diag([obj.precisionToVariance(obj.prior_precision_retention), ...
            obj.precisionToVariance(obj.prior_precision_drift)]);
    end
    covar = obj.regularizeCovariance(covar);
end
