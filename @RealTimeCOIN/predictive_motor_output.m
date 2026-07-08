function u = predictive_motor_output(obj, q)
%PREDICTIVE_MOTOR_OUTPUT Expected next observation given the upcoming cue.
%   u = predictive_motor_output(obj, q) returns the model's point prediction
%   of the next state feedback, conditioned on the upcoming cue label q. It is
%   a read-only one-step prediction from the current posterior (it draws no
%   random numbers and does not mutate particle state), so it may be called
%   between observe_q(q) and observe_y(y).
%
%   If q is omitted (or []) the pending cue staged by observe_q is used. For
%   the scalar model (state_dim == 1) u is a scalar; for the multi-dimensional
%   model u is an N-by-1 vector (the mean returned by
%   predictive_feedback_moments).
    arguments
        obj (1, 1) RealTimeCOIN
        q double {mustBeScalarOrEmpty, mustBeInteger, mustBeNonnegative} = []
    end

    if isempty(q)
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
