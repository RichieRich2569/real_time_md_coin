function results = validate_ensemble_vs_coin_blindB(varargin)
%VALIDATE_ENSEMBLE_VS_COIN Run-averaged RealTimeCOINEnsemble vs offline COIN.m.
%
%   Scientific validator (blind, author B). It checks that the run-averaged
%   motor-output trajectory produced by RealTimeCOINEnsemble on a perturbation
%   paradigm matches the run-averaged trajectory produced by the reference
%   COIN.m (COIN.simulate_COIN) with a matching number of runs, within a stated
%   Monte-Carlo tolerance. Returns a struct of metrics and pass flags (it does
%   not assert like a unit test), mirroring the other validation/ scripts.
%
%   Fairness of the comparison (see docs/SPEC_ensemble.md). COIN.m GENERATES
%   its feedback internally, state_feedback = perturbation + sensory_noise +
%   motor_noise, whereas the ensemble is FED the feedback y. To make both see
%   the same effective feedback we:
%     * set COIN's sigma_motor_noise = 0 and sigma_sensory_noise to a tiny value
%       (default 1e-3) so each COIN run's generated feedback is, to within that
%       tiny noise, equal to the perturbation;
%     * feed the ensemble the perturbation itself as the observed feedback y;
%     * give the ensemble members the SAME hyperparameters as COIN (including
%       the same tiny sigma_sensory_noise), so the two only differ by the
%       Monte-Carlo error of averaging R independent process-noise realizations.
%
%   Alignment assumption (underdetermined by the spec, documented per task).
%   COIN stores, at trial t, the pre-feedback predicted motor output for trial
%   t. The ensemble's Phase-1 query surface exposes only motor_output(ens) read
%   out AFTER each trial (there is no cue-conditioned predictive query). The
%   exact trial-phase correspondence between "ensemble motor_output after trial
%   t" and "COIN stored motor_output(t)" is therefore not fixed by the spec, so
%   this validator evaluates BOTH natural alignments -- lag 0 (same index) and
%   lag 1 (ensemble leads COIN by one trial) -- reports both, and gates on the
%   better-fitting one.

ip = inputParser;
addParameter(ip, 'Runs', 16);
addParameter(ip, 'Trials', 90);
addParameter(ip, 'Particles', 80);
addParameter(ip, 'Seed', 4201);
addParameter(ip, 'MaxCores', 0);
addParameter(ip, 'SigmaSensory', 1e-3);
addParameter(ip, 'MakePlots', false);
addParameter(ip, 'Strict', false);
parse(ip, varargin{:});
cfg = ip.Results;

rootDir = fileparts(fileparts(mfilename('fullpath')));
addpath(rootDir);
addpath(fullfile(rootDir, 'validation'));

T = cfg.Trials;
n1 = floor(T / 3);
n2 = floor(T / 3);
perturbations = [zeros(1, n1), 0.4 * ones(1, n2), -0.2 * ones(1, T - n1 - n2)];
cues = ones(1, T);
cues(floor(T / 2):end) = 2;

sigmaR = cfg.SigmaSensory;
sigmaMotor = 0;
maxContexts = 4;

% ------------------------------------------------------------------ %
% Reference: offline COIN.m with R runs, tiny observation noise.
% ------------------------------------------------------------------ %
rng(cfg.Seed);
old = COIN;
old.perturbations = perturbations;
old.cues = cues;
old.runs = cfg.Runs;
old.particles = cfg.Particles;
old.max_contexts = maxContexts;
old.store = {'motor_output'};
old.sigma_sensory_noise = sigmaR;
old.sigma_motor_noise = sigmaMotor;
old.plot_state_feedback = false;
old.max_cores = 0;

S = old.simulate_COIN;

coinRuns = zeros(cfg.Runs, T);
for r = 1:cfg.Runs
    runData = S.runs{r};
    if isfield(runData, 'stored') && isfield(runData.stored, 'motor_output')
        mo = runData.stored.motor_output;
    else
        mo = runData.motor_output;
    end
    coinRuns(r, :) = mo(:)';
end
coinAvg = mean(coinRuns, 1, 'omitnan');

% ------------------------------------------------------------------ %
% Ensemble: R members sharing COIN's hyperparameters, FED the perturbation.
% ------------------------------------------------------------------ %
memberParams = {'num_particles', cfg.Particles, 'max_contexts', maxContexts, ...
    'gamma_context', old.gamma_context, 'alpha_context', old.alpha_context, ...
    'rho_context', old.rho_context, 'gamma_cue', old.gamma_cue, ...
    'alpha_cue', old.alpha_cue, ...
    'prior_mean_retention', old.prior_mean_retention, ...
    'prior_precision_retention', old.prior_precision_retention, ...
    'prior_mean_drift', 0, 'prior_precision_drift', old.prior_precision_drift, ...
    'sigma_process_noise', old.sigma_process_noise, ...
    'sigma_sensory_noise', sigmaR, 'sigma_motor_noise', sigmaMotor};

ens = RealTimeCOINEnsemble('runs', cfg.Runs, 'seed', cfg.Seed, ...
    'max_cores', cfg.MaxCores, memberParams{:});

ySeq = perturbations;                 % feed the perturbation as observed feedback
tr = ens.simulate(cues, ySeq);
ensMotor = tr.motor_output(:)';

% ------------------------------------------------------------------ %
% Compare run-averaged trajectories under both natural alignments.
% ------------------------------------------------------------------ %
[rmse0, corr0] = local_fit(coinAvg, ensMotor);            % lag 0
[rmse1, corr1] = local_fit(coinAvg(2:T), ensMotor(1:T-1)); % lag 1 (ens leads)

if rmse1 < rmse0
    rmseBest = rmse1; corrBest = corr1; alignment = 1;
else
    rmseBest = rmse0; corrBest = corr0; alignment = 0;
end

% Monte-Carlo tolerances. Both estimators average R independent runs whose
% feedback is (to within sigmaR ~ 1e-3) the same deterministic perturbation, so
% they estimate the same posterior-mean trajectory and differ only by particle-
% filter / run-averaging Monte-Carlo error. The gates are looser than the
% implementation-vs-implementation gate in validate_original_coin_monte_carlo
% (RMSE 0.03) because here the two use INDEPENDENT RNG streams and independent
% run sets rather than a shared feedback stream.
thresholds = struct();
thresholds.rmse = 0.08;
thresholds.correlation = 0.85;

checks = struct();
checks.rmse = rmseBest < thresholds.rmse;
checks.correlation = corrBest > thresholds.correlation;
[passed, checks] = validation_pass_summary(checks);

results = struct();
results.rmse = rmseBest;
results.correlation = corrBest;
results.rmse_lag0 = rmse0;
results.rmse_lag1 = rmse1;
results.correlation_lag0 = corr0;
results.correlation_lag1 = corr1;
results.chosen_alignment_lag = alignment;
results.coin_run_average = coinAvg;
results.ensemble_run_average = ensMotor;
results.perturbations = perturbations;
results.cues = cues;
results.thresholds = thresholds;
results.checks = checks;
results.passed = passed;
results.config = cfg;

fprintf(['Ensemble vs COIN: RMSE %.4f (lag %d), corr %.3f ' ...
    '[lag0 rmse %.4f, lag1 rmse %.4f]\n'], ...
    rmseBest, alignment, corrBest, rmse0, rmse1);

if cfg.MakePlots
    figure('Name', 'Ensemble vs COIN run-averaged motor output');
    plot(coinAvg, 'k-', 'LineWidth', 1.2); hold on;
    plot(ensMotor, 'r--', 'LineWidth', 1.2);
    plot(perturbations, 'b:', 'LineWidth', 1.0);
    xlabel('trial'); ylabel('motor output');
    legend({'COIN run-average', 'Ensemble run-average', 'perturbation'}, ...
        'Location', 'best');
    title(sprintf('RMSE %.4f (lag %d), corr %.3f', rmseBest, alignment, corrBest));
end

if cfg.Strict && ~passed
    error('validate_ensemble_vs_coin:Failed', ...
        'Ensemble vs COIN validation failed.');
end
end

function [rmse, corrValue] = local_fit(a, b)
%LOCAL_FIT RMSE and Pearson correlation over the finite overlap of a and b.
a = a(:);
b = b(:);
valid = isfinite(a) & isfinite(b);
if ~any(valid)
    rmse = NaN;
    corrValue = NaN;
    return;
end
rmse = sqrt(mean((a(valid) - b(valid)).^2));
if sum(valid) > 1 && std(a(valid)) > 0 && std(b(valid)) > 0
    c = corrcoef(a(valid), b(valid));
    corrValue = c(1, 2);
else
    corrValue = NaN;
end
end
