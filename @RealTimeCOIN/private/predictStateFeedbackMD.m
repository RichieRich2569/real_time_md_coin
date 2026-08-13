function predictStateFeedbackMD(obj)
%PREDICTSTATEFEEDBACKMD Multivariate predictive observation distribution.
%
%   Multi-dimensional counterpart of predictStateFeedback.m. With the
%   identity observation map y = s + b + v, v ~ N(0, R), the predictive
%   observation (feedback) distribution per context is
%
%       y_pred = s_{i|i-1} + b
%       S      = P_{i|i-1} + R           (innovation covariance)
%
%   S is reused both as the resampling likelihood covariance and as the
%   denominator of the Kalman gain in updateBeliefAboutStatesMD.

    R = obj.observationNoiseCov();

    obj.D.state_feedback_mean = obj.D.state_mean + obj.D.bias;
    % Add the shared observation noise R to every context/particle innovation
    % covariance via implicit expansion: R is N-by-N and state_cov is
    % N-by-N-by-Cmax-by-P, so R broadcasts across the trailing context/particle
    % dims - identical values to repmat(R,1,1,Cmax,P) without materialising it.
    obj.D.state_feedback_cov = obj.D.state_cov + R;
end
