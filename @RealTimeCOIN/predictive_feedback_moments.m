function [mu, Sigma] = predictive_feedback_moments(obj, q)
%PREDICTIVE_FEEDBACK_MOMENTS One-step predictive observation moments.
%
%   [mu, Sigma] = predictive_feedback_moments(obj, q) returns the mean and
%   covariance of the next observation (state feedback) given the optional
%   upcoming cue q, marginalised over contexts and particles. It performs a
%   read-only one-step prediction from the current posterior (it does NOT
%   mutate particle state), so it can be called between observe_q(q) and
%   observe_y(y) to obtain the model's belief about the imminent feedback.
%
%   For the scalar model (state_dim == 1) mu and Sigma are scalars; for the
%   multi-dimensional model mu is N-by-1 and Sigma is N-by-N. This is the
%   in-class, MD-capable counterpart of the scalar-only validation helper
%   validation_predictive_feedback_moments and is used by the multi-
%   dimensional Kalman validation to compute predictive calibration.
    arguments
        obj (1, 1) RealTimeCOIN
        q double {mustBeScalarOrEmpty, mustBeInteger, mustBeNonnegative} = []
    end

    Cmax = obj.max_contexts + 1;
    P = obj.num_particles;

    % --- Context mixture weights: transition prior, optionally x cue likelihood ---
    prior = zeros(Cmax, P);
    for p = 1:P
        prior(:, p) = obj.D.local_transition_matrix(obj.D.context(p), :, p)';
    end
    prior = obj.normalizeColumns(prior);
    if isempty(q)
        weights = prior;
    else
        qCol = min(q, size(obj.D.local_cue_matrix, 2));
        cueLik = squeeze(obj.D.local_cue_matrix(:, qCol, :));
        if P == 1
            cueLik = cueLik(:);
        end
        weights = obj.normalizeColumns(prior .* cueLik);
    end

    if obj.state_dim == 1
        [mu, Sigma] = scalarMoments(obj, weights, Cmax, P);
    else
        [mu, Sigma] = multiMoments(obj, weights, Cmax, P);
    end
end

function [mu, v] = scalarMoments(obj, weights, Cmax, P)
%SCALARMOMENTS One-step predictive feedback moments for the scalar model.
    stateMean = obj.D.retention .* obj.D.state_filtered_mean + obj.D.drift;
    stateVar = obj.D.retention.^2 .* obj.D.state_filtered_var + obj.sigma_process_noise^2;
    for p = 1:P
        novel = min(obj.D.C(p) + 1, Cmax);
        if obj.D.C(p) < obj.max_contexts
            a = obj.D.retention(novel, p);
            d = obj.D.drift(novel, p);
            stateMean(novel, p) = d ./ max(1 - a, eps);
            stateVar(novel, p) = obj.sigma_process_noise^2 ./ max(1 - a.^2, eps);
        end
    end
    feedbackMean = stateMean + obj.D.bias;
    feedbackVar = stateVar + obj.observationVariance();

    mu = sum(weights .* feedbackMean, 'all') ./ P;
    second = sum(weights .* (feedbackVar + feedbackMean.^2), 'all') ./ P;
    v = max(second - mu.^2, 0);
end

function [mu, Sigma] = multiMoments(obj, weights, Cmax, P)
%MULTIMOMENTS One-step predictive feedback moments for the MD model.
    N = obj.state_dim;
    Q = obj.processNoiseCov();
    R = obj.observationNoiseCov();

    mu = zeros(N, 1);
    second = zeros(N, N);
    for p = 1:P
        novel = min(obj.D.C(p) + 1, Cmax);
        for c = 1:Cmax
            w = weights(c, p) / P;
            if w == 0
                continue;
            end
            A = obj.D.Theta(:, 1:N, c, p);
            d = obj.D.Theta(:, N+1, c, p);
            if c == novel && obj.D.C(p) < obj.max_contexts
                sPred = obj.stationaryStateMeanMD(A, d);
                PPred = obj.stationaryStateCovMD(A, Q);
            else
                sPred = A * obj.D.state_filtered_mean(:, c, p) + d;
                PPred = A * obj.D.state_filtered_cov(:, :, c, p) * A' + Q;
            end
            fbMean = sPred + obj.D.bias(:, c, p);
            fbCov = PPred + R;
            mu = mu + w .* fbMean;
            second = second + w .* (fbCov + (fbMean * fbMean'));
        end
    end
    Sigma = second - (mu * mu');
    Sigma = (Sigma + Sigma') ./ 2;
end

function mustBeScalarOrEmpty(x)
    if ~(isempty(x) || isscalar(x))
        error('RealTimeCOIN:InvalidCue', ...
            'q must be empty or a scalar cue label.');
    end
end
