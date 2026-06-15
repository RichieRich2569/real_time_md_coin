function [mu, covar] = localDynamicsDistribution(obj, local, p)
    if isfield(obj.D, 'dynamics_mean') && ...
            size(obj.D.dynamics_mean, 2) >= local && size(obj.D.dynamics_mean, 3) >= p
        mu = obj.D.dynamics_mean(:,local,p);
    else
        mu = [obj.D.retention(local,p); obj.D.drift(local,p)];
    end

    if isfield(obj.D, 'dynamics_covar') && ...
            size(obj.D.dynamics_covar, 3) >= local && size(obj.D.dynamics_covar, 4) >= p
        covar = obj.D.dynamics_covar(:,:,local,p);
    else
        covar = diag([obj.precisionToVariance(obj.prior_precision_retention), ...
            obj.precisionToVariance(obj.prior_precision_drift)]);
    end
    covar = obj.regularizeCovariance(covar);
end
