function u = predictive_motor_output(obj, q)
    if nargin < 2
        q = obj.pending_q;
    end
    qLabel = peekCueLabel(obj, q);
    [W, M, ~] = previewPredictiveFeedback(obj, qLabel);
    u = sum(W .* M, 'all') ./ obj.num_particles;
end
