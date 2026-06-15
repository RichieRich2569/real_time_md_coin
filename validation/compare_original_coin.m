function results = compare_original_coin(varargin)
%COMPARE_ORIGINAL_COIN Probabilistic comparison with the original COIN.m.
%
%   The original COIN implementation generates feedback internally. This
%   script runs it once, feeds the generated feedback stream into
%   RealTimeCOIN, and compares predictive motor outputs.

ip = inputParser;
addParameter(ip, 'Trials', 100);
addParameter(ip, 'Particles', 100);
addParameter(ip, 'Seed', 2001);
parse(ip, varargin{:});
cfg = ip.Results;

rng(cfg.Seed);
rootDir = fileparts(fileparts(mfilename('fullpath')));
addpath(rootDir);

perturbations = [zeros(1, floor(cfg.Trials/3)), ...
    0.4 * ones(1, floor(cfg.Trials/3)), ...
    -0.2 * ones(1, cfg.Trials - 2*floor(cfg.Trials/3))];
cues = ones(1, cfg.Trials);
cues(floor(cfg.Trials/2):end) = 2;

old = COIN;
old.perturbations = perturbations;
old.cues = cues;
old.runs = 1;
old.particles = cfg.Particles;
old.max_contexts = 4;
old.store = {'motor_output', 'state_feedback'};
old.sigma_motor_noise = 0;
old.plot_state_feedback = false;

S = old.simulate_COIN;
y = S.runs{1}.state_feedback;
if isfield(S.runs{1}, 'stored')
    oldMotor = S.runs{1}.stored.motor_output;
else
    oldMotor = S.runs{1}.motor_output;
end
oldMotor = oldMotor(:)';

rng(cfg.Seed);
rt = RealTimeCOIN('num_particles', cfg.Particles, 'max_contexts', 4, ...
    'gamma_context', old.gamma_context, 'alpha_context', old.alpha_context, ...
    'rho_context', old.rho_context, 'gamma_cue', old.gamma_cue, ...
    'alpha_cue', old.alpha_cue, 'prior_mean_retention', old.prior_mean_retention, ...
    'prior_precision_retention', old.prior_precision_retention, ...
    'prior_mean_drift', 0, 'prior_precision_drift', old.prior_precision_drift, ...
    'sigma_process_noise', old.sigma_process_noise, ...
    'sigma_sensory_noise', old.sigma_sensory_noise, 'sigma_motor_noise', old.sigma_motor_noise);

rtMotor = zeros(1, cfg.Trials);
for t = 1:cfg.Trials
    rtMotor(t) = rt.predictive_motor_output(cues(t));
    rt.observe_q(cues(t));
    rt.observe_y(y(t));
end

valid = isfinite(oldMotor) & isfinite(rtMotor);
rmse = sqrt(mean((oldMotor(valid) - rtMotor(valid)).^2));
if sum(valid) > 1
    corrValue = corr(oldMotor(valid)', rtMotor(valid)');
else
    corrValue = NaN;
end

results = struct();
results.rmse_motor_output = rmse;
results.correlation_motor_output = corrValue;
results.original_motor_output = oldMotor;
results.realtime_motor_output = rtMotor;
results.feedback = y;
results.config = cfg;

fprintf('Original/RT motor output RMSE: %.4f, correlation: %.3f\n', rmse, corrValue);
end
