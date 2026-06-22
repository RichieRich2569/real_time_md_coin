function results = validate_stress_cases(varargin)
%VALIDATE_STRESS_CASES Hand-designed behavioural stress checks.
%
%   These cases are not proofs of calibration.  They are deliberately
%   interpretable probes of qualitative behaviour: stable data should not
%   create many contexts, abrupt changes should increase contextual
%   uncertainty or create a new context, A-B-A data should tend to reuse an
%   old context, and sensory-noise settings should change posterior
%   uncertainty in the expected direction.

ip = inputParser;
addParameter(ip, 'Trials', 90);
addParameter(ip, 'Particles', 100);
addParameter(ip, 'Seed', 1501);
addParameter(ip, 'Strict', false);
parse(ip, varargin{:});
cfg = ip.Results;

rng(cfg.Seed);

single = run_case(constant_signal(cfg.Trials, 0), ones(1, cfg.Trials), cfg, 1);
abrupt = run_case(block_signal(cfg.Trials, [0 0.4]), block_cues(cfg.Trials, [1 2]), cfg, 2);
aba = run_case(block_signal(cfg.Trials, [0 0.4 0]), block_cues(cfg.Trials, [1 2 1]), cfg, 3);
uninformative = run_case(block_signal(cfg.Trials, [0 0.35]), random_cues(cfg.Trials), cfg, 4);
misleading = run_case(block_signal(cfg.Trials, [0 0.35]), block_cues(cfg.Trials, [2 1]), cfg, 5);
highNoise = run_noise_case(cfg, 0.12, 6);
lowNoise = run_noise_case(cfg, 0.015, 7);
cap = run_case(block_signal(cfg.Trials, [0 0.3 -0.2 0.45]), ...
    block_cues(cfg.Trials, [1 2 3 4]), cfg, 8, 2);

thresholds = struct();
thresholds.single_context_mean_count = 1.5;
thresholds.abrupt_creates_context = 1.5;
thresholds.cap_max_contexts = 2;
thresholds.high_noise_uncertainty_ratio = 1.25;

checks = struct();
checks.single_context_mean_count = single.mean_context_count <= thresholds.single_context_mean_count;
checks.abrupt_creates_context = abrupt.max_context_count >= thresholds.abrupt_creates_context;
checks.cap_respected = cap.max_context_count <= thresholds.cap_max_contexts;
checks.noise_uncertainty = highNoise.mean_state_variance > ...
    thresholds.high_noise_uncertainty_ratio * lowNoise.mean_state_variance;
[passed, checks] = validation_pass_summary(checks);

results = struct();
results.single_context_no_cue_change = single;
results.abrupt_perturbation_switch = abrupt;
results.aba_repeated_context = aba;
results.uninformative_cues = uninformative;
results.misleading_cues = misleading;
results.high_sensory_noise = highNoise;
results.low_sensory_noise = lowNoise;
results.max_context_cap = cap;
results.thresholds = thresholds;
results.checks = checks;
results.passed = passed;
results.config = cfg;

fprintf('Stress cases: single mean K %.2f, abrupt max K %.0f, high/low var ratio %.2f\n', ...
    single.mean_context_count, abrupt.max_context_count, ...
    highNoise.mean_state_variance / max(lowNoise.mean_state_variance, eps));

if cfg.Strict && ~passed
    error('validate_stress_cases:Failed', 'Stress-case validation failed.');
end
end

function out = run_case(signal, cues, cfg, seedOffset, maxContexts)
if nargin < 5
    maxContexts = 5;
end
rng(cfg.Seed + seedOffset);
coin = RealTimeCOIN('num_particles', cfg.Particles, 'max_contexts', maxContexts, ...
    'prior_mean_retention', 0.92, 'prior_precision_retention', 300, ...
    'prior_mean_drift', 0.0, 'prior_precision_drift', 100, ...
    'sigma_process_noise', 0.025, 'sigma_sensory_noise', 0.04, ...
    'sigma_motor_noise', 0);

T = numel(signal);
contextCount = zeros(1, T);
topContext = zeros(1, T);
stateVariance = zeros(1, T);
predictionError = zeros(1, T);

for t = 1:T
    y = signal(t) + 0.04 * randn;
    pred = coin.predictive_motor_output(cues(t));
    coin.observe_q(cues(t));
    coin.observe_y(y);
    diag = coin.diagnostics();
    resp = mean(diag.responsibilities, 2);
    [~, topContext(t)] = max(resp);
    contextCount(t) = diag.C;
    stateVariance(t) = mean(diag.raw.state_filtered_var(diag.raw.i_observed));
    predictionError(t) = abs(y - pred);
end

out = struct();
out.mean_context_count = mean(contextCount);
out.max_context_count = max(contextCount);
out.final_context_count = contextCount(end);
out.mean_state_variance = mean(stateVariance, 'omitnan');
out.mean_prediction_error = mean(predictionError, 'omitnan');
out.context_count = contextCount;
out.top_context = topContext;
out.reuses_initial_context_in_final_block = final_block_reuses_initial(topContext);
end

function out = run_noise_case(cfg, sensoryNoise, seedOffset)
rng(cfg.Seed + seedOffset);
coin = RealTimeCOIN('num_particles', cfg.Particles, 'max_contexts', 3, ...
    'sigma_process_noise', 0.025, 'sigma_sensory_noise', sensoryNoise, ...
    'sigma_motor_noise', 0);
T = cfg.Trials;
stateVariance = zeros(1, T);
contextCount = zeros(1, T);
for t = 1:T
    y = 0.1 * sin(t / 12) + sensoryNoise * randn;
    coin.observe_q(1);
    coin.observe_y(y);
    diag = coin.diagnostics();
    stateVariance(t) = mean(diag.raw.state_filtered_var(diag.raw.i_observed));
    contextCount(t) = diag.C;
end
out = struct();
out.mean_context_count = mean(contextCount);
out.max_context_count = max(contextCount);
out.final_context_count = contextCount(end);
out.mean_state_variance = mean(stateVariance, 'omitnan');
out.context_count = contextCount;
end

function x = constant_signal(T, value)
x = value * ones(1, T);
end

function x = block_signal(T, values)
edges = round(linspace(1, T + 1, numel(values) + 1));
x = zeros(1, T);
for i = 1:numel(values)
    x(edges(i):edges(i+1)-1) = values(i);
end
end

function q = block_cues(T, values)
q = round(block_signal(T, values));
end

function q = random_cues(T)
q = 1 + double(rand(1, T) > 0.5);
end

function tf = final_block_reuses_initial(topContext)
T = numel(topContext);
firstBlock = topContext(1:floor(T/3));
lastBlock = topContext(2*floor(T/3)+1:end);
tf = mode(lastBlock) == mode(firstBlock);
end
