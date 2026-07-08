function predictStateFeedback(obj)
%PREDICTSTATEFEEDBACK Scalar predictive observation (feedback) distribution.
%   predictStateFeedback(obj) maps the predicted latent state to the predictive
%   feedback distribution under the observation model y = s + b + v, with
%   v ~ N(0, observationVariance):
%
%       state_feedback_mean = state_mean + bias
%       state_feedback_var  = state_var  + observationVariance
%
%   observationVariance is sigma_sensory_noise^2 + sigma_motor_noise^2. The
%   resulting variance is the scalar innovation variance reused as the
%   resampling likelihood variance and as the Kalman-gain denominator in
%   updateBeliefAboutStates. Scalar counterpart of predictStateFeedbackMD.
%
%   Writes obj.D.state_feedback_mean and obj.D.state_feedback_var
%   (each (max_contexts+1)-by-num_particles).

    obj.D.state_feedback_mean = obj.D.state_mean + obj.D.bias;
    obj.D.state_feedback_var = obj.D.state_var + obj.observationVariance();
end
