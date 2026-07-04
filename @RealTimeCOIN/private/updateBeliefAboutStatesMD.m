function updateBeliefAboutStatesMD(obj, y, obsMask)
% =========================================================================
% MATHEMATICAL NOTE: MULTIVARIATE KALMAN MEASUREMENT UPDATE
% =========================================================================
% With the identity observation model y = s + b + v, v ~ N(0, R), the
% innovation and its covariance for the active context are
%     y_tilde = y - (s_pred + b),   S = P_pred + R.
% The Kalman gain, posterior mean and posterior covariance are
%     K      = P_pred * S^-1
%     s_post = s_pred + K * y_tilde
%     P_post = (I - K) * P_pred
% At N == 1 this is exactly the scalar update in updateBeliefAboutStates.m
% (K = predVar/(predVar+obsVar)). Inactive contexts inherit the prediction
% unchanged, since no measurement informs them on this trial.
% =========================================================================
%
%UPDATEBELIEFABOUTSTATESMD Multivariate Kalman update for the active context.

    N = obj.state_dim;
    P = obj.num_particles;
    if nargin < 3 || isempty(obsMask)
        obsMask = ~isnan(y(:));
    else
        obsMask = obsMask(:);
    end

    % Inactive contexts: posterior == prior prediction.
    obj.D.state_filtered_mean = obj.D.state_mean;
    obj.D.state_filtered_cov = obj.D.state_cov;
    if isempty(y) || ~any(obsMask)
        return;
    end

    R = obj.observationNoiseCov();
    yv = y(:);
    I = eye(N);
    obsIdx = find(obsMask);
    R_obs = R(obsIdx, obsIdx);
    for p = 1:P
        c = obj.D.context(p);
        Pp = obj.D.state_cov(:, :, c, p);
        sp = obj.D.state_mean(:, c, p);
        yhat = obj.D.state_feedback_mean(obsIdx, c, p);   % s_pred + b
        S = Pp(obsIdx, obsIdx) + R_obs;

        % K = Pp * inv(S) computed via a right solve for stability.
        K = Pp(:, obsIdx) / S;
        innovation = yv(obsIdx) - yhat;
        sf = sp + K * innovation;
        KH = zeros(N, N);
        KH(:, obsIdx) = K;
        IKH = I - KH;
        Pf = IKH * Pp * IKH' + K * R_obs * K';
        Pf = (Pf + Pf') ./ 2;     % enforce symmetry against round-off

        obj.D.state_filtered_mean(:, c, p) = sf;
        obj.D.state_filtered_cov(:, :, c, p) = Pf;
    end
end
