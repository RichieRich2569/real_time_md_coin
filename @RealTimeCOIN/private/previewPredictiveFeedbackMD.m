function [W, M, Cov] = previewPredictiveFeedbackMD(obj, q)
%PREVIEWPREDICTIVEFEEDBACKMD One-step predictive feedback mixture (MD model).
%
%   Multi-dimensional counterpart of previewPredictiveFeedback.m. Performs a
%   read-only one-step look-ahead from the current posterior, returning the
%   per-particle/per-context Gaussian-mixture components of the next
%   observation given the optional upcoming cue q (q = [] marginalises the
%   cue out):
%       W   : (max_contexts+1)-by-P normalised mixture weights
%       M   : N-by-(max_contexts+1)-by-P  predictive feedback means
%       Cov : N-by-N-by-(max_contexts+1)-by-P predictive feedback covariances
%
%   Each component mirrors the multivariate Kalman one-step prediction in
%   predictStateFeedbackMD / predictive_feedback_moments: with the identity
%   observation map y = s + b + v (v ~ N(0,R)) and dynamics Theta = [A | d],
%       s_pred = A s_{i-1|i-1} + d,    P_pred = A P_{i-1|i-1} A' + Q,
%       fbMean = s_pred + b,           fbCov  = P_pred + R.
%   The first not-yet-instantiated ("novel") context uses the stationary
%   prediction (stationaryStateMeanMD / stationaryStateCovMD) instead.
    arguments
        obj (1, 1) RealTimeCOIN
        q double {mustBeScalarOrEmpty, mustBeInteger, mustBeFinite, mustBeNonnegative} = []
    end

    N = obj.state_dim;
    Cmax = obj.max_contexts + 1;
    P = obj.num_particles;
    Q = obj.processNoiseCov();
    R = obj.observationNoiseCov();

    % --- Context mixture weights: transition prior, optionally x cue likelihood ---
    prior = zeros(Cmax, P);
    for p = 1:P
        prior(:, p) = obj.D.local_transition_matrix(obj.D.context(p), :, p)';
    end
    prior = obj.normalizeColumns(prior);
    if isempty(q)
        W = prior;
    else
        qCol = min(q, size(obj.D.local_cue_matrix, 2));
        cueLik = squeeze(obj.D.local_cue_matrix(:, qCol, :));
        if P == 1
            cueLik = cueLik(:);
        end
        W = obj.normalizeColumns(prior .* cueLik);
    end

    % --- Per-context predictive feedback Gaussian components ---
    M = zeros(N, Cmax, P);
    Cov = zeros(N, N, Cmax, P);
    for p = 1:P
        novel = min(obj.D.C(p) + 1, Cmax);
        for c = 1:Cmax
            A = obj.D.Theta(:, 1:N, c, p);
            d = obj.D.Theta(:, N+1, c, p);
            if c == novel && obj.D.C(p) < obj.max_contexts
                sPred = obj.stationaryStateMeanMD(A, d);
                PPred = obj.stationaryStateCovMD(A, Q);
            else
                sPred = A * obj.D.state_filtered_mean(:, c, p) + d;
                PPred = A * obj.D.state_filtered_cov(:, :, c, p) * A' + Q;
            end
            M(:, c, p) = sPred + obj.D.bias(:, c, p);
            Cov(:, :, c, p) = PPred + R;
        end
    end
end

function mustBeScalarOrEmpty(x)
    if ~isscalar(x) && ~isempty(x)
        error('Input must be a scalar or empty.');
    end
end
