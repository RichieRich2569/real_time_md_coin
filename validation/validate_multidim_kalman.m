function results = validate_multidim_kalman(varargin)
%VALIDATE_MULTIDIM_KALMAN Independent multivariate Kalman reference validation.
%
%   In the one-context, N-dimensional, linear-Gaussian case the COIN state
%   model reduces to a multivariate Kalman filter:
%
%       s_t = A s_{t-1} + d + w_t,   w_t ~ N(0, Q)
%       y_t = s_t + v_t,             v_t ~ N(0, R).
%
%   With very precise dynamics priors and max_contexts == 1, RealTimeCOIN
%   should be doing only the multivariate Kalman predict-update recursion.
%   Correlated (full) Q and R are used so the matrix Kalman gain and the
%   Cholesky likelihood are genuinely exercised. This is an external
%   mathematical reference, not a comparison against the original COIN.m.
%
%   Calibration is assessed with a multivariate probability-integral
%   transform: the Mahalanobis distance of each innovation under the
%   predictive feedback covariance is chi-square_N distributed when the model
%   is calibrated, so chi2cdf(mahalanobis, N) should be Uniform(0,1).

ip = inputParser;
addParameter(ip, 'Trials', 120);
addParameter(ip, 'Particles', 300);
addParameter(ip, 'Dim', 2);
addParameter(ip, 'Seed', 2201);
addParameter(ip, 'MakePlots', false);
addParameter(ip, 'Strict', false);
parse(ip, varargin{:});
cfg = ip.Results;

rng(cfg.Seed);

N = cfg.Dim;
a = 0.82;
drift = 0.035;
A = a * eye(N);
d = drift * ones(N, 1);

% Correlated process/observation noise (symmetric positive definite).
Q = 0.025^2 * (eye(N) + 0.3 * (ones(N) - eye(N)));
R = 0.05^2 * (eye(N) - 0.2 * (ones(N) - eye(N)));
Q = (Q + Q') / 2;
R = (R + R') / 2;

coin = RealTimeCOIN('num_particles', cfg.Particles, 'max_contexts', 1, 'state_dim', N, ...
    'prior_mean_retention', a, 'prior_precision_retention', 1e12, ...
    'prior_mean_drift', drift, 'prior_precision_drift', 1e12, ...
    'process_noise_covariance', Q, 'observation_noise_covariance', R);

Lq = chol(Q, 'lower');
Lr = chol(R, 'lower');

% Kalman filter and model initialised at the shared stationary distribution
% (P0 = Q/(1-a^2) since A = a I).
m = d ./ (1 - a);
Pcov = Q ./ (1 - a^2);
s = m;

kalmanMean = zeros(N, cfg.Trials);
rtMean = zeros(N, cfg.Trials);
varRel = zeros(1, cfg.Trials);
rtPit = zeros(1, cfg.Trials);

for t = 1:cfg.Trials
    s = A * s + d + Lq * randn(N, 1);
    y = s + Lr * randn(N, 1);

    mPred = A * m + d;
    PPred = A * Pcov * A' + Q;
    S = PPred + R;
    kalmanMean(:, t) = mPred;

    coin.observe_q(1);
    [mu, Sigma] = coin.predictive_feedback_moments(1);
    rtMean(:, t) = mu;
    varRel(t) = norm(Sigma - S, 'fro') ./ max(norm(S, 'fro'), eps);

    innovation = y - mu;
    mahalanobis = innovation' * (Sigma \ innovation);
    rtPit(t) = gammainc(mahalanobis / 2, N / 2);   % chi-square_N CDF

    K = PPred / S;
    m = mPred + K * (y - mPred);
    Pcov = (eye(N) - K) * PPred;
    Pcov = (Pcov + Pcov') ./ 2;

    coin.observe_y(y);
end

meanRmse = sqrt(mean((rtMean(:) - kalmanMean(:)).^2));
varRelError = median(varRel);
feedbackKs = validation_uniform_ks(rtPit);

thresholds = struct();
thresholds.mean_rmse = 0.05;
thresholds.variance_relative_error = 0.35;
thresholds.feedback_ks = 0.15;

checks = struct();
checks.mean_rmse = meanRmse < thresholds.mean_rmse;
checks.variance_relative_error = varRelError < thresholds.variance_relative_error;
checks.feedback_ks = feedbackKs < thresholds.feedback_ks;
[passed, checks] = validation_pass_summary(checks);

results = struct();
results.dim = N;
results.predictive_mean_rmse = meanRmse;
results.predictive_variance_relative_error = varRelError;
results.feedback_ks = feedbackKs;
results.realtime_predictive_mean = rtMean;
results.kalman_predictive_mean = kalmanMean;
results.feedback_p_values = rtPit;
results.thresholds = thresholds;
results.checks = checks;
results.passed = passed;
results.config = cfg;

fprintf('Multi-dim Kalman (N=%d): mean RMSE %.4f, variance rel. error %.3f, PIT KS %.3f\n', ...
    N, meanRmse, varRelError, feedbackKs);

if cfg.MakePlots
    figure('Name', 'Multi-dimensional Kalman validation');
    tiledlayout(1, 2);
    nexttile; plot(kalmanMean', 'k-'); hold on; plot(rtMean', 'r--');
    title('Predictive mean (per dimension)'); legend({'Kalman', 'RealTimeCOIN'});
    nexttile; histogram(rtPit, 20, 'Normalization', 'probability');
    title('Feedback PIT (chi-square)');
end

if cfg.Strict && ~passed
    error('validate_multidim_kalman:Failed', 'Multi-dimensional Kalman validation failed.');
end
end
