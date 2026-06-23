function u = predictive_motor_output(obj, q)
%PREDICTIVE_MOTOR_OUTPUT Expected next observation given the upcoming cue.
%   Scalar for state_dim == 1; an N-by-1 vector for the multi-dimensional
%   model (the mean of predictive_feedback_moments).
    if nargin < 2
        q = obj.pending_q;
    end
    qLabel = peekCueLabel(obj, q);
    if obj.state_dim > 1
        u = obj.predictive_feedback_moments(qLabel);
        return;
    end
    [W, M, ~] = previewPredictiveFeedback(obj, qLabel);
    u = sum(W .* M, 'all') ./ obj.num_particles;
end
