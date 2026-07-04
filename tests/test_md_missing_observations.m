function test_md_missing_observations
%TEST_MD_MISSING_OBSERVATIONS Multi-dimensional NaN feedback handling.

rng(11);

% All-NaN MD feedback should be exactly the same channel trial as [] when
% the random stream and model configuration are identical.
cfg = {'state_dim', 2, 'num_particles', 40, 'max_contexts', 3, ...
    'infer_bias', true, ...
    'process_noise_covariance', [1.0e-4, 2.0e-5; 2.0e-5, 1.1e-4], ...
    'observation_noise_covariance', [4.0e-4, -1.0e-4; -1.0e-4, 5.0e-4]};

rng(12);
coinEmpty = RealTimeCOIN(cfg{:});
coinEmpty.observe_q(3);
coinEmpty.observe_y([]);
dEmpty = coinEmpty.diagnostics();

rng(12);
coinNaN = RealTimeCOIN(cfg{:});
coinNaN.observe_q(3);
coinNaN.observe_y([NaN; NaN]);
dNaN = coinNaN.diagnostics();

assert(coinNaN.Trial == 1, 'All-NaN trial did not advance the trial counter');
assertClose(dNaN.raw.responsibilities, dEmpty.raw.responsibilities, 1e-12, ...
    'All-NaN responsibilities differ from empty feedback');
assertClose(dNaN.raw.state_filtered_mean, dEmpty.raw.state_filtered_mean, 1e-12, ...
    'All-NaN filtered means differ from empty feedback');
assertClose(dNaN.raw.bias, dEmpty.raw.bias, 1e-12, ...
    'All-NaN bias samples differ from empty feedback');
assertFiniteMD(dNaN.raw);

% Mixed missingness should use the observed coordinate and keep every model
% field finite, including inferred bias statistics.
rng(13);
coinMixed = RealTimeCOIN(cfg{:});
coinMixed.observe_y([0.1; NaN]);
coinMixed.observe_y([NaN; -0.2]);
dMixed = coinMixed.diagnostics();
assertFiniteMD(dMixed.raw);
assert(abs(sum(dMixed.raw.responsibilities(:)) - coinMixed.num_particles) < 1e-9, ...
    'Mixed-NaN responsibilities are not normalized per particle');

validatePartialObservationKalman();
validateCorrelatedUnobservedUpdate();
end

function validatePartialObservationKalman
N = 2;
a = 0.8;
drift = 0.03;
A = a * eye(N);
d = drift * ones(N, 1);
Q = [1.0e-4, 3.0e-5; 3.0e-5, 1.2e-4];
R = [4.0e-4, -1.0e-4; -1.0e-4, 5.0e-4];
T = 18;
masks = logical([1 1; 1 0; 0 1; 0 0; 1 1; 1 0; 0 1; 0 0; ...
    1 1; 1 0; 0 1; 0 0; 1 1; 1 0; 0 1; 0 0; 1 1; 1 0])';

rng(14);
Lq = chol(Q, 'lower');
Lr = chol(R, 'lower');
s = d ./ (1 - a);
yFull = zeros(N, T);
for t = 1:T
    s = A * s + d + Lq * randn(N, 1);
    yFull(:, t) = s + Lr * randn(N, 1);
end

m = d ./ (1 - a);
Pcov = Q ./ (1 - a^2);

rng(15);
coin = RealTimeCOIN('num_particles', 500, 'max_contexts', 1, 'state_dim', N, ...
    'prior_mean_retention', a, 'prior_precision_retention', 1e12, ...
    'prior_mean_drift', drift, 'prior_precision_drift', 1e12, ...
    'process_noise_covariance', Q, 'observation_noise_covariance', R);

kfPredMean = zeros(N, T);
rtPredMean = zeros(N, T);

for t = 1:T
    mPred = A * m + d;
    PPred = A * Pcov * A' + Q;
    kfPredMean(:, t) = mPred;

    coin.observe_q(1);
    [mu, Sigma] = coin.predictive_feedback_moments(1);
    rtPredMean(:, t) = mu;
    assert(min(eig((Sigma + Sigma') ./ 2)) > -1e-9, ...
        'Partial-observation predictive covariance is not PSD');

    obsMask = masks(:, t);
    yObs = yFull(:, t);
    yObs(~obsMask) = NaN;
    if any(obsMask)
        obsIdx = find(obsMask);
        S = PPred(obsIdx, obsIdx) + R(obsIdx, obsIdx);
        K = PPred(:, obsIdx) / S;
        innovation = yFull(obsIdx, t) - mPred(obsIdx);
        KH = zeros(N, N);
        KH(:, obsIdx) = K;
        m = mPred + K * innovation;
        Pcov = (eye(N) - KH) * PPred * (eye(N) - KH)' + K * R(obsIdx, obsIdx) * K';
        Pcov = (Pcov + Pcov') ./ 2;
    else
        m = mPred;
        Pcov = PPred;
    end
    coin.observe_y(yObs);
end

rmse = sqrt(mean((rtPredMean(:) - kfPredMean(:)).^2));
assert(rmse < 0.06, ...
    sprintf('Partial-observation MD Kalman predictive means differ too much (RMSE %.4f)', rmse));
end

function validateCorrelatedUnobservedUpdate
N = 2;
R = [4.0e-4, -1.0e-4; -1.0e-4, 5.0e-4];
Q = [1.0e-4, 7.0e-5; 7.0e-5, 1.2e-4];
cfg = {'num_particles', 1, 'max_contexts', 1, 'state_dim', N, ...
    'prior_mean_retention', 0.8, 'prior_precision_retention', 1e12, ...
    'prior_mean_drift', 0.0, 'prior_precision_drift', 1e12, ...
    'process_noise_covariance', Q, 'observation_noise_covariance', R};

rng(16);
coinChannel = RealTimeCOIN(cfg{:});
coinChannel.observe_y([]);
dChannel = coinChannel.diagnostics();

rng(16);
coinPartial = RealTimeCOIN(cfg{:});
coinPartial.observe_y([0.25; NaN]);
dPartial = coinPartial.diagnostics();

deltaUnobserved = abs(dPartial.raw.state_filtered_mean(2, 1, 1) - ...
    dChannel.raw.state_filtered_mean(2, 1, 1));
assert(deltaUnobserved > 1e-8, ...
    'Unobserved dimension did not update differently from a channel trial');
end

function assertFiniteMD(D)
fields = {'responsibilities', 'state_filtered_mean', 'state_filtered_cov', ...
    'state_mean', 'state_cov', 'state_feedback_mean', 'state_feedback_cov', ...
    'bias', 'bias_info_ss', 'bias_precision_ss', 'probability_state_feedback'};
for i = 1:numel(fields)
    f = fields{i};
    if isfield(D, f)
        assert(all(isfinite(D.(f)(:))), sprintf('Field %s contains nonfinite values', f));
    end
end
end

function assertClose(actual, expected, tol, msg)
err = max(abs(actual(:) - expected(:)));
assert(err <= tol, sprintf('%s (max abs err %.3g)', msg, err));
end
