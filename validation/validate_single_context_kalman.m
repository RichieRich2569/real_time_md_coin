function results = validate_single_context_kalman(varargin)
%VALIDATE_SINGLE_CONTEXT_KALMAN Independent Kalman reference validation.
%
%   In the one-context, scalar, linear-Gaussian case the COIN state model
%   reduces to a Kalman filter:
%
%       s_t = a s_{t-1} + d + eps_t,    eps_t ~ N(0, sigma_Q^2)
%       y_t = s_t + eta_t,              eta_t ~ N(0, sigma_R^2).
%
%   If the retention a and drift d priors are made very precise and
%   max_contexts is one, RealTimeCOIN should only be doing the Kalman
%   predict-update recursion.  This test is therefore an external
%   mathematical reference, not a comparison against the original COIN.m.

ip = inputParser;
addParameter(ip, 'Trials', 120);
addParameter(ip, 'Particles', 250);
addParameter(ip, 'Seed', 1101);
addParameter(ip, 'MakePlots', false);
addParameter(ip, 'Strict', false);
parse(ip, varargin{:});
cfg = ip.Results;

rng(cfg.Seed);

a = 0.82;
d = 0.035;
sigmaQ = 0.025;
sigmaR = 0.05;

coin = RealTimeCOIN('num_particles', cfg.Particles, 'max_contexts', 1, ...
    'prior_mean_retention', a, 'prior_precision_retention', 1e12, ...
    'prior_mean_drift', d, 'prior_precision_drift', 1e12, ...
    'sigma_process_noise', sigmaQ, 'sigma_sensory_noise', sigmaR, ...
    'sigma_motor_noise', 0);

m = d / (1 - a);
P = sigmaQ^2 / (1 - a^2);
s = m;

kalmanMean = zeros(1, cfg.Trials);
kalmanVar = zeros(1, cfg.Trials);
rtMean = zeros(1, cfg.Trials);
rtVar = zeros(1, cfg.Trials);
rtPit = zeros(1, cfg.Trials);
analyticPit = zeros(1, cfg.Trials);
yTrace = zeros(1, cfg.Trials);

for t = 1:cfg.Trials
    s = a * s + d + sigmaQ * randn;
    y = s + sigmaR * randn;
    yTrace(t) = y;

    [mPred, yVar, m, P] = validation_kalman_reference(m, P, a, d, sigmaQ^2, sigmaR^2, y);

    kalmanMean(t) = mPred;
    kalmanVar(t) = yVar;
    [rtMean(t), rtVar(t)] = validation_predictive_feedback_moments(coin, 1);
    rtPit(t) = coin.predictive_state_feedback_cdf(y, 1);
    analyticPit(t) = RealTimeCOIN.normal_cdf(y, mPred, yVar);

    coin.observe_q(1);
    coin.observe_y(y);
end

meanRmse = sqrt(mean((rtMean - kalmanMean).^2));
varRelError = median(abs(rtVar - kalmanVar) ./ max(kalmanVar, eps));
feedbackKs = validation_uniform_ks(rtPit);
analyticKs = validation_uniform_ks(analyticPit);

% Gates are shared verbatim with validate_multidim_kalman so the scalar and
% multivariate Kalman references are held to one standard.  mean_rmse (0.05)
% and variance_relative_error (0.35) bound agreement with the analytic
% Kalman moments; feedback_ks (0.15) is a single-stream PIT gate and is
% looser than the p_values_extended feedback gate (0.08) because that
% validator pools many datasets and so estimates the KS statistic far more
% tightly.
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
results.predictive_mean_rmse = meanRmse;
results.predictive_variance_relative_error = varRelError;
results.feedback_ks = feedbackKs;
results.analytic_feedback_ks = analyticKs;
results.realtime_predictive_mean = rtMean;
results.kalman_predictive_mean = kalmanMean;
results.realtime_predictive_variance = rtVar;
results.kalman_predictive_variance = kalmanVar;
results.feedback_p_values = rtPit;
results.feedback = yTrace;
results.thresholds = thresholds;
results.checks = checks;
results.passed = passed;
results.config = cfg;

fprintf('Single-context Kalman: mean RMSE %.4f, variance rel. error %.3f, PIT KS %.3f\n', ...
    meanRmse, varRelError, feedbackKs);

if cfg.MakePlots
    figure('Name', 'Single-context Kalman validation');
    tiledlayout(1, 2);
    nexttile; plot(kalmanMean, 'k-'); hold on; plot(rtMean, 'r--');
    title('Predictive mean'); legend({'Kalman', 'RealTimeCOIN'});
    nexttile; histogram(rtPit, 20, 'Normalization', 'probability');
    title('Feedback PIT');
end

if cfg.Strict && ~passed
    error('validate_single_context_kalman:Failed', 'Single-context Kalman validation failed.');
end
end
