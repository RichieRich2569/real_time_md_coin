function results = validate_context_recovery(varargin)
%VALIDATE_CONTEXT_RECOVERY Recovery of known latent contexts.
%
%   Synthetic data gives us the hidden context c_t, but mixture labels are
%   arbitrary: an inferred context labelled "3" may correspond to the true
%   context labelled "1".  We therefore score context recovery only after
%   finding the label permutation that minimizes mismatch with the true
%   labels.  This separates genuine inference failure from harmless label
%   switching.
%
%   Seed averaging.  The recovery accuracy sits close to its 0.65 gate, so a
%   single-seed run is seed-sensitive: the same code can pass or fail purely
%   from which particle-filter noise draw it happened to use.  To remove that
%   flip we run the experiment over a small set of seeds derived from the
%   'Seed' parameter (Seed + (0:NumSeeds-1)), compute the recovery metrics on
%   each, and gate on the ACROSS-SEED AVERAGE of accuracy, true-context
%   posterior mass, and mean recovery lag.  The thresholds, the printed
%   summary line, and the results struct shape are unchanged; the top-level
%   metric fields now hold the seed-averaged values, per-seed detail is
%   exposed under results.per_seed, and the trace fields (cues/feedback/
%   inferred/matched contexts) come from the first (representative) seed.

ip = inputParser;
addParameter(ip, 'Trials', 120);
addParameter(ip, 'Particles', 150);
addParameter(ip, 'Seed', 1401);
addParameter(ip, 'NumSeeds', 5);
addParameter(ip, 'Strict', false);
parse(ip, varargin{:});
cfg = ip.Results;

seeds = cfg.Seed + (0:cfg.NumSeeds - 1);

% Run each seed independently and collect the per-seed metric structs.
% Seed the array from the first run so every element shares the same fields.
perSeed = run_one_seed(seeds(1), cfg);
for i = 2:numel(seeds)
    perSeed(i) = run_one_seed(seeds(i), cfg);
end

% Across-seed averages: these are what the gates act on.
avgAccuracy = mean([perSeed.accuracy]);
avgPosterior = mean([perSeed.mean_posterior_true_context]);
avgInferredCount = mean([perSeed.mean_inferred_context_count]);

lagMatrix = vertcat(perSeed.recovery_lags);          % numSeeds x numSwitches
avgLags = mean(lagMatrix, 1, 'omitnan');
avgMeanLag = mean([perSeed.mean_recovery_lag], 'omitnan');

% Gate rationale: these are qualitative recovery gates, not calibration
% gates, so they are intentionally generous.  context_accuracy (0.65) is the
% best-relabelled hard-assignment agreement; posterior_true_context (0.45)
% is the softer average posterior mass on the true context; and
% mean_recovery_lag (<=20 trials) bounds how quickly the filter re-locks
% after a context switch.  Averaging over NumSeeds seeds (see header) removes
% the cross-seed flips that used to occur right at the accuracy boundary.
thresholds = struct();
thresholds.context_accuracy = 0.65;
thresholds.posterior_true_context = 0.45;
thresholds.mean_recovery_lag = 20;

checks = struct();
checks.context_accuracy = avgAccuracy > thresholds.context_accuracy;
checks.posterior_true_context = avgPosterior > thresholds.posterior_true_context;
checks.mean_recovery_lag = avgMeanLag <= thresholds.mean_recovery_lag;
[passed, checks] = validation_pass_summary(checks);

rep = perSeed(1);   % representative seed for trace/diagnostic fields

results = struct();
results.context_accuracy = avgAccuracy;
results.mean_posterior_true_context = avgPosterior;
results.recovery_lags = avgLags;
results.mean_recovery_lag = avgMeanLag;
results.mean_inferred_context_count = avgInferredCount;
results.true_context = rep.true_context;
results.inferred_context = rep.inferred_context;
results.matched_context = rep.matched_context;
results.posterior_true_context = rep.posterior_true_context;
results.feedback = rep.feedback;
results.cues = rep.cues;
results.thresholds = thresholds;
results.checks = checks;
results.passed = passed;
results.per_seed = perSeed;
results.seeds = seeds;
results.config = cfg;

fprintf('Context recovery: accuracy %.3f, true-context posterior %.3f, mean lag %.1f trials\n', ...
    avgAccuracy, avgPosterior, avgMeanLag);

if cfg.Strict && ~passed
    error('validate_context_recovery:Failed', 'Context recovery validation failed.');
end
end

function out = run_one_seed(seed, cfg)
%RUN_ONE_SEED Execute one context-recovery experiment at a fixed seed.
rng(seed);

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

out = struct();
out.seed = seed;
out.accuracy = accuracy;
out.mean_posterior_true_context = mean(posteriorTrue);
out.recovery_lags = lags;
out.mean_recovery_lag = mean(lags, 'omitnan');
out.mean_inferred_context_count = mean(inferredCount);
out.true_context = trueContext;
out.inferred_context = inferred;
out.matched_context = mapped;
out.posterior_true_context = posteriorTrue;
out.feedback = y;
out.cues = q;
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
