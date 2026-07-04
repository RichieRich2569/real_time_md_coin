function sampleStatesMD(obj, y, obsMask)
%SAMPLESTATESMD Sample latent state trajectories (RTS smoother, multivariate).
%
%   Multi-dimensional counterpart of sampleStates.m. Two latent quantities
%   are sampled per context per particle to drive parameter learning:
%
%   1. The lag state s_{i-1} via the Rauch-Tung-Striebel smoother gain
%          J = P_{i-1|i-1} A' P_{i|i-1}^-1
%      so that
%          mean_s  = s_{i-1|i-1} + J (s_{i|i} - s_{i|i-1})
%          cov_s   = P_{i-1|i-1} + J (P_{i|i} - P_{i|i-1}) J'.
%      At N == 1 J reduces to the scalar gain g = a P_{prev}/P_pred.
%
%   2. The current state s_i: for inactive contexts (or when no observation
%      is available) it is drawn from the one-step dynamics prior
%      N(A s_{i-1} + d, Q). For the active context the dynamics prior is
%      combined in information form with the observation N(y - b, R):
%          postPrec = Q^-1 + R^-1,    postMean = postCov (Q^-1 m_dyn + R^-1 (y-b)).
%      This is the multivariate generalisation of the scalar posterior in
%      sampleStates.m.

    N = obj.state_dim;
    Cmax = obj.max_contexts + 1;
    P = obj.num_particles;
    Q = obj.processNoiseCov();
    R = obj.observationNoiseCov();
    Qi = obj.safeInverse(Q);
    if nargin < 3 || isempty(obsMask)
        obsMask = ~isnan(y(:));
    else
        obsMask = obsMask(:);
    end
    hasObservation = ~isempty(y) && any(obsMask);
    if hasObservation
        obsIdx = find(obsMask);
        R_obs = R(obsIdx, obsIdx);
        Ri_obs = obj.safeInverse(R_obs);
        obsPrecision = zeros(N, N);
        obsPrecision(obsIdx, obsIdx) = Ri_obs;
    end

    obj.D.previous_x_dynamics = zeros(N, Cmax, P);
    obj.D.x_dynamics = zeros(N, Cmax, P);
    if hasObservation
        yv = y(:);
    end

    for p = 1:P
        active = obj.D.context(p);
        for c = 1:Cmax
            A = obj.D.Theta(:, 1:N, c, p);
            d = obj.D.Theta(:, N+1, c, p);

            % --- 1. Smoother sample of the lag state s_{i-1} ---
            Ppred = obj.D.state_cov(:, :, c, p);
            Pprev = obj.D.previous_state_filtered_cov(:, :, c, p);
            J = Pprev * A' * obj.safeInverse(Ppred);
            meanS = obj.D.previous_state_filtered_mean(:, c, p) + ...
                J * (obj.D.state_filtered_mean(:, c, p) - obj.D.state_mean(:, c, p));
            covS = Pprev + J * (obj.D.state_filtered_cov(:, :, c, p) - Ppred) * J';
            covS = (covS + covS') ./ 2;
            sPrev = drawGaussian(obj, meanS, covS);
            obj.D.previous_x_dynamics(:, c, p) = sPrev;

            % --- 2. Forward sample of the current state s_i ---
            dynMean = A * sPrev + d;
            if ~hasObservation || c ~= active
                postMean = dynMean;
                postCov = Q;
            else
                postCov = obj.safeInverse(Qi + obsPrecision);
                postCov = (postCov + postCov') ./ 2;
                bias = obj.D.bias(:, c, p);
                obsInfo = zeros(N, 1);
                obsInfo(obsIdx) = Ri_obs * (yv(obsIdx) - bias(obsIdx));
                postMean = postCov * (Qi * dynMean + obsInfo);
            end
            obj.D.x_dynamics(:, c, p) = drawGaussian(obj, postMean, postCov);
        end
    end

    % Active-context sampled state (used for the bias residual) and its index.
    obj.D.x_bias = zeros(N, P);
    for p = 1:P
        obj.D.x_bias(:, p) = obj.D.x_dynamics(:, obj.D.context(p), p);
    end
end

function x = drawGaussian(obj, mu, Sigma)
%DRAWGAUSSIAN Draw a single sample from N(mu, Sigma) via Cholesky.
    if all(Sigma(:) == 0)
        x = mu(:);
        return;
    end
    [L, ~] = obj.choljitter(Sigma);
    x = mu(:) + L * randn(numel(mu), 1);
end
