function [predMean, predFeedbackCov, postMean, postCov] = validation_kalman_reference(m, P, A, d, Q, R, y)
%VALIDATION_KALMAN_REFERENCE One step of the linear-Gaussian Kalman recursion.
%
%   [predMean, predFeedbackCov, postMean, postCov] = ...
%       VALIDATION_KALMAN_REFERENCE(m, P, A, d, Q, R, y)
%   runs a single predict-update step of the multivariate Kalman filter for
%   the one-context COIN state model
%
%       s_t = A s_{t-1} + d + w_t,   w_t ~ N(0, Q)
%       y_t = s_t + v_t,             v_t ~ N(0, R).
%
%   Inputs are the prior state mean M and covariance P, the dynamics A and
%   drift D, the process and observation covariances Q and R, and the
%   observation Y.  Outputs are the one-step predictive state mean PREDMEAN,
%   the predictive feedback covariance PREDFEEDBACKCOV (= A P A' + Q + R),
%   and the posterior (filtered) mean POSTMEAN and covariance POSTCOV.
%
%   The scalar single-context case is the N == 1 specialisation of this
%   recursion, so both validate_single_context_kalman and
%   validate_multidim_kalman share this reference implementation.  Keeping
%   the recursion in one place ensures the scalar and multivariate
%   validators cannot drift apart.

predMean = A * m + d;
predStateCov = A * P * A' + Q;
predFeedbackCov = predStateCov + R;

% Kalman gain K = predStateCov * inv(predFeedbackCov); solved as a right
% division so no explicit inverse is formed.
K = predStateCov / predFeedbackCov;

postMean = predMean + K * (y - predMean);
postCov = predStateCov - K * predStateCov;      % = (I - K) predStateCov
postCov = (postCov + postCov') / 2;             % re-symmetrise (no-op when scalar)
end
