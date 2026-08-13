function [mu, v] = state_moments(obj)
%STATE_MOMENTS Moments of the run-pooled predictive-state mixture.
%   [mu, v] = state_moments(obj) returns the mean and (co)variance of the pooled
%   mixture that gives each of the R members weight 1/R at the current trial.
%   The covariance is the law-of-total-(co)variance combination of the per-run
%   moments (see poolMoments), NOT a naive average of the per-run covariances:
%       mu = (1/R) sum_k mu_k
%       v  = (1/R) sum_k ( v_k + mu_k mu_k' ) - mu mu'.
%
%   Scalar model (state_dim == 1): mu and v are scalars (v floored at 0).
%   Multi-dimensional model: mu is N-by-1 and v is the symmetric N-by-N
%   covariance. Read-only: draws no randomness.
%
%   See also RealTimeCOIN/state_moments, MOTOR_OUTPUT.
    arguments
        obj (1, 1) RealTimeCOINEnsemble
    end
    R = obj.runs;
    mus = cell(1, R);
    vs = cell(1, R);
    for k = 1:R
        [mk, vk] = obj.members{k}.state_moments();
        mus{k} = mk(:);
        vs{k} = vk;
    end
    [mu, v] = poolMoments(mus, vs, obj.state_dim_);
end
