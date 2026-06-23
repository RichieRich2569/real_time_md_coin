function p = predictive_state_feedback_cdf(obj, y, q)
    obj.assertScalarOnly('predictive_state_feedback_cdf');
    if nargin < 3
        q = obj.pending_q;
    end
    qLabel = peekCueLabel(obj, q);
    [W, M, V] = previewPredictiveFeedback(obj, qLabel);
    p = sum(W .* obj.normal_cdf(y, M, V), 'all') ./ obj.num_particles;
    p = min(max(p, 0), 1);
end
