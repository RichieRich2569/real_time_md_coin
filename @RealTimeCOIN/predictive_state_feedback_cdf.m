function p = predictive_state_feedback_cdf(obj, y, q)
%PREDICTIVE_STATE_FEEDBACK_CDF Predictive CDF of the next feedback at y.
%
%   For the scalar model (state_dim == 1) returns the scalar predictive
%   probability P(Y <= y) of the next feedback given the optional upcoming cue
%   q (defaults to the pending cue). For the multi-dimensional model y is an
%   N-vector and the return is an N-by-1 vector of marginal predictive CDFs,
%   p_j = P(Y_j <= y_j), which is the standard per-dimension probability-
%   integral-transform used for calibration. Each marginal of the predictive
%   Gaussian mixture is itself a 1-D Gaussian mixture, so this reuses the
%   scalar normal_cdf and reduces exactly to the scalar result at N == 1.

    if nargin < 3
        q = obj.pending_q;
    end
    qLabel = peekCueLabel(obj, q);

    if obj.state_dim == 1
        [W, M, V] = previewPredictiveFeedback(obj, qLabel);
        p = sum(W .* obj.normal_cdf(y, M, V), 'all') ./ obj.num_particles;
        p = min(max(p, 0), 1);
        return;
    end

    N = obj.state_dim;
    y = y(:);
    if numel(y) ~= N
        error('RealTimeCOIN:FeedbackDimensionMismatch', ...
            ['predictive_state_feedback_cdf expects y to have %d elements for ', ...
             'state_dim == %d; received %d.'], N, N, numel(y));
    end
    [W, M, Cov] = previewPredictiveFeedbackMD(obj, qLabel);
    Cmax = obj.max_contexts + 1;
    p = zeros(N, 1);
    for j = 1:N
        Mj = reshape(M(j, :, :), Cmax, obj.num_particles);
        Vj = reshape(Cov(j, j, :, :), Cmax, obj.num_particles);
        p(j) = sum(W .* obj.normal_cdf(y(j), Mj, Vj), 'all') ./ obj.num_particles;
    end
    p = min(max(p, 0), 1);
end
