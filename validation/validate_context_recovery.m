function results = validate_context_recovery(varargin)
%VALIDATE_CONTEXT_RECOVERY Recovery of known latent contexts.
%
%   Synthetic data gives us the hidden context c_t, but mixture labels are
%   arbitrary: an inferred context labelled "3" may correspond to the true
%   context labelled "1".  We therefore score context recovery only after
%   finding the label permutation that minimizes mismatch with the true
%   labels.  This separates genuine inference failure from harmless label
%   switching.

ip = inputParser;
addParameter(ip, 'Trials', 120);
addParameter(ip, 'Particles', 150);
addParameter(ip, 'Seed', 1401);
addParameter(ip, 'Strict', false);
parse(ip, varargin{:});
cfg = ip.Results;

rng(cfg.Seed);

trueContext = synthetic_context_sequence(cfg.Trials);
a = [0.92 0.90];
d = [0.000 0.045];
sigmaQ = 0.025;
sigmaR = 0.045;
cueProb = [0.88 0.12; 0.12 0.88];

coin = RealTimeCOIN('num_particles', cfg.Particles, 'max_contexts', 4, ...
    'prior_mean_retention', 0.9, 'prior_precision_retention', 200, ...
    'prior_mean_drift', 0.015, 'prior_precision_drift', 80, ...
    'sigma_process_noise', sigmaQ, 'sigma_sensory_noise', sigmaR, ...
    'sigma_motor_noise', 0);

s = 0;
q = zeros(1, cfg.Trials);
y = zeros(1, cfg.Trials);
inferred = zeros(1, cfg.Trials);
Kmax = coin.max_contexts + 1;
respHistory = zeros(Kmax, cfg.Trials);
inferredCount = zeros(1, cfg.Trials);

for t = 1:cfg.Trials
    c = trueContext(t);
    q(t) = validation_sample_categorical(cueProb(c, :));
    s = a(c) * s + d(c) + sigmaQ * randn;
    y(t) = s + sigmaR * randn;

    coin.observe_q(q(t));
    coin.observe_y(y(t));

    diag = coin.diagnostics();
    resp = mean(diag.responsibilities, 2)';
    respHistory(1:numel(resp), t) = resp;
    [~, inferred(t)] = max(resp);
    inferredCount(t) = diag.C;
end

[mapped, accuracy, mapping] = validation_best_label_map(trueContext, inferred);
posteriorTrue = zeros(1, cfg.Trials);
for t = 1:cfg.Trials
    for inferredLabel = 1:Kmax
        if isKey(mapping, inferredLabel) && mapping(inferredLabel) == trueContext(t)
            posteriorTrue(t) = posteriorTrue(t) + respHistory(inferredLabel, t);
        end
    end
end

switches = find(diff(trueContext) ~= 0) + 1;
lags = recovery_lags(mapped, trueContext, switches);

% Gate rationale: these are qualitative recovery gates, not calibration
% gates, so they are intentionally generous.  context_accuracy (0.65) is the
% best-relabelled hard-assignment agreement; posterior_true_context (0.45)
% is the softer average posterior mass on the true context; and
% mean_recovery_lag (<=20 trials) bounds how quickly the filter re-locks
% after a context switch.  These can be seed-sensitive near the boundary;
% widen the margin or average over a seed loop for publication runs.
thresholds = struct();
thresholds.context_accuracy = 0.65;
thresholds.posterior_true_context = 0.45;
thresholds.mean_recovery_lag = 20;

checks = struct();
checks.context_accuracy = accuracy > thresholds.context_accuracy;
checks.posterior_true_context = mean(posteriorTrue) > thresholds.posterior_true_context;
checks.mean_recovery_lag = mean(lags, 'omitnan') <= thresholds.mean_recovery_lag;
[passed, checks] = validation_pass_summary(checks);

results = struct();
results.context_accuracy = accuracy;
results.mean_posterior_true_context = mean(posteriorTrue);
results.recovery_lags = lags;
results.mean_recovery_lag = mean(lags, 'omitnan');
results.mean_inferred_context_count = mean(inferredCount);
results.true_context = trueContext;
results.inferred_context = inferred;
results.matched_context = mapped;
results.posterior_true_context = posteriorTrue;
results.feedback = y;
results.cues = q;
results.thresholds = thresholds;
results.checks = checks;
results.passed = passed;
results.config = cfg;

fprintf('Context recovery: accuracy %.3f, true-context posterior %.3f, mean lag %.1f trials\n', ...
    accuracy, results.mean_posterior_true_context, results.mean_recovery_lag);

if cfg.Strict && ~passed
    error('validate_context_recovery:Failed', 'Context recovery validation failed.');
end
end

function c = synthetic_context_sequence(T)
third = floor(T / 3);
c = [ones(1, third), 2 * ones(1, third), ones(1, T - 2 * third)];
end

function lags = recovery_lags(mapped, trueContext, switches)
lags = NaN(size(switches));
for i = 1:numel(switches)
    t0 = switches(i);
    target = trueContext(t0);
    idx = find(mapped(t0:end) == target, 1, 'first');
    if ~isempty(idx)
        lags(i) = idx - 1;
    end
end
end
