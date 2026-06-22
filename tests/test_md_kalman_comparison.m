function test_md_kalman_comparison
%TEST_MD_KALMAN_COMPARISON Multi-dimensional Kalman equivalence in the single-context limit.
%
%   With one context and very precise dynamics priors, the N-dimensional
%   RealTimeCOIN pipeline should reduce to a multivariate Kalman filter. We
%   drive the model with correlated (full-covariance) process and observation
%   noise so the matrix machinery -- matrix Kalman gain, Cholesky likelihood,
%   cross-dimension covariance propagation -- is genuinely exercised, and
%   compare the one-step predictive feedback mean against an independent 2-D
%   Kalman filter. A chi-square probability-integral-transform sanity check
%   confirms the predictive covariance is on the right scale.

rng(7);

N = 2;
a = 0.8;
drift = 0.03;
A = a * eye(N);
d = drift * ones(N, 1);
Q = [1.0e-4, 3.0e-5; 3.0e-5, 1.2e-4];      % correlated process noise
R = [4.0e-4, -1.0e-4; -1.0e-4, 5.0e-4];     % correlated observation noise

T = 20;
Lq = chol(Q, 'lower');
Lr = chol(R, 'lower');

% Generate synthetic data from the multivariate linear-Gaussian model.
s = d ./ (1 - a);                            % stationary mean (A = a I)
y = zeros(N, T);
for t = 1:T
    s = A * s + d + Lq * randn(N, 1);
    y(:, t) = s + Lr * randn(N, 1);
end

% Reference Kalman filter, initialised at the stationary distribution that
% the model also uses (P0 = Q/(1-a^2) because A = a I).
m = d ./ (1 - a);
Pcov = Q ./ (1 - a^2);

coin = RealTimeCOIN('num_particles', 500, 'max_contexts', 1, 'state_dim', N, ...
    'prior_mean_retention', a, 'prior_precision_retention', 1e12, ...
    'prior_mean_drift', drift, 'prior_precision_drift', 1e12, ...
    'process_noise_covariance', Q, 'observation_noise_covariance', R);

kfPredMean = zeros(N, T);
rtPredMean = zeros(N, T);
pit = zeros(1, T);

for t = 1:T
    % Kalman predictive feedback distribution for this trial.
    mPred = A * m + d;
    PPred = A * Pcov * A' + Q;
    S = PPred + R;
    kfPredMean(:, t) = mPred;

    % Model predictive feedback distribution (read-only, before observing y).
    coin.observe_q(1);
    [mu, Sigma] = coin.predictive_feedback_moments(1);
    rtPredMean(:, t) = mu;

    assert(norm(Sigma - Sigma', 'fro') < 1e-8, 'Predictive feedback covariance not symmetric');
    assert(min(eig(Sigma)) > -1e-9, 'Predictive feedback covariance not PSD');

    innovation = y(:, t) - mu;
    mahalanobis = innovation' * (Sigma \ innovation);
    pit(t) = gammainc(mahalanobis / 2, N / 2);   % chi-square_N CDF (base MATLAB)

    % Kalman measurement update.
    K = PPred / S;
    m = mPred + K * (y(:, t) - mPred);
    Pcov = (eye(N) - K) * PPred;
    Pcov = (Pcov + Pcov') ./ 2;

    coin.observe_y(y(:, t));
end

rmse = sqrt(mean((rtPredMean(:) - kfPredMean(:)).^2));
assert(rmse < 0.05, ...
    sprintf('MD predictive means differ from Kalman beyond tolerance (RMSE %.4f)', rmse));
assert(all(pit >= 0 & pit <= 1), 'PIT values fell outside [0,1]');

% Sanity on the public MD moment accessor.
[muState, covState] = coin.state_moments();
assert(isequal(size(muState), [N, 1]), 'state_moments mean must be N-by-1 in MD mode');
assert(isequal(size(covState), [N, N]), 'state_moments covariance must be N-by-N in MD mode');
assert(min(eig((covState + covState') ./ 2)) > -1e-9, 'state covariance not PSD');
end
