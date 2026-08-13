function results = run_validation(varargin)
%RUN_VALIDATION Run the compact RealTimeCOIN validation suite.
%
%   results = RUN_VALIDATION() runs a deliberately compact scientific
%   validation suite.  These scripts report metrics and pass flags rather
%   than acting like fast unit tests.  Increase the arguments in individual
%   validators for publication-scale calibration runs.

ip = inputParser;
addParameter(ip, 'Profile', 'compact');
addParameter(ip, 'MakePlots', false);
addParameter(ip, 'Seed', 1001);
addParameter(ip, 'Strict', false);
parse(ip, varargin{:});
cfg = ip.Results;

rootDir = fileparts(fileparts(mfilename('fullpath')));
addpath(rootDir);
addpath(fullfile(rootDir, 'validation'));

switch lower(cfg.Profile)
    case 'compact'
        suite = compact_config(cfg);
    otherwise
        error('run_validation:UnknownProfile', ...
            'Unknown validation profile "%s".', cfg.Profile);
end

results = struct();
results.config = cfg;

% Each validator is run inside run_stage so that one failing validator
% records passed=false plus the captured error and the suite continues,
% rather than an uncaught error tearing down the independent validators
% that follow it.
%
% Every sub-validator is called with 'Strict', false (never cfg.Strict)
% on purpose: forwarding Strict makes a sub-validator error() internally on
% failure, which run_stage's try/catch would collapse into a bare
% {passed,errored,error} struct, discarding the full metric struct we want
% to inspect.  With Strict=false the sub-validators always return their
% complete metrics (with passed=false on failure), and strictness is
% enforced once at the SUITE level below via the results.passed gate.
results.single_context_kalman = run_stage('single_context_kalman', @() ...
    validate_single_context_kalman( ...
        'Trials', suite.kalman_trials, 'Particles', suite.kalman_particles, ...
        'Seed', cfg.Seed + 10, 'MakePlots', cfg.MakePlots, 'Strict', false));
results.multidim_kalman = run_stage('multidim_kalman', @() ...
    validate_multidim_kalman( ...
        'Trials', suite.kalman_trials, 'Particles', suite.kalman_particles, ...
        'Dim', 2, 'Seed', cfg.Seed + 15, 'MakePlots', cfg.MakePlots, 'Strict', false));
results.p_values_extended = run_stage('p_values_extended', @() ...
    validate_p_values_extended( ...
        'NumDatasets', suite.pit_datasets, 'Trials', suite.pit_trials, ...
        'Particles', suite.pit_particles, 'Seed', cfg.Seed + 20, ...
        'MakePlots', cfg.MakePlots, 'Strict', false));
results.original_coin_monte_carlo = run_stage('original_coin_monte_carlo', @() ...
    validate_original_coin_monte_carlo( ...
        'Seeds', cfg.Seed + (30:34), 'Trials', suite.original_trials, ...
        'Particles', suite.original_particles, 'Strict', false));
results.particle_convergence = run_stage('particle_convergence', @() ...
    validate_particle_convergence( ...
        'Particles', suite.convergence_particles, 'Trials', suite.convergence_trials, ...
        'NumDatasets', suite.convergence_datasets, 'Seed', cfg.Seed + 40, ...
        'Strict', false));
results.context_recovery = run_stage('context_recovery', @() ...
    validate_context_recovery( ...
        'Trials', suite.context_trials, 'Particles', suite.context_particles, ...
        'Seed', cfg.Seed + 50, 'Strict', false));
results.stress_cases = run_stage('stress_cases', @() ...
    validate_stress_cases( ...
        'Trials', suite.stress_trials, 'Particles', suite.stress_particles, ...
        'Seed', cfg.Seed + 60, 'Strict', false));
results.performance = run_stage('performance', @() ...
    benchmark_realtimecoin( ...
        'Trials', suite.benchmark_trials, 'Particles', suite.benchmark_particles, ...
        'Seed', cfg.Seed + 70));

% Compatibility aliases retained deliberately: existing scripts/notebooks
% (and the MATLAB Project file registry) refer to these older field names,
% so they are kept as views onto the current validators rather than removed.
results.p_values = results.p_values_extended;
results.original_comparison = results.original_coin_monte_carlo;

results.passed = results.single_context_kalman.passed && ...
    results.multidim_kalman.passed && ...
    results.p_values_extended.passed && ...
    results.original_coin_monte_carlo.passed && ...
    results.particle_convergence.passed && ...
    results.context_recovery.passed && ...
    results.stress_cases.passed;

fprintf('Compact validation suite passed: %d\n', results.passed);

if cfg.Strict && ~results.passed
    error('run_validation:Failed', 'One or more validation checks failed.');
end
end

function out = run_stage(name, fn)
%RUN_STAGE Execute one validator, isolating failures from the rest of the suite.
try
    out = fn();
catch err
    fprintf(2, 'Validator "%s" errored: %s\n', name, err.message);
    out = struct();
    out.passed = false;
    out.errored = true;
    out.error = err;
end
end

function suite = compact_config(~)
suite = struct();
suite.kalman_trials = 80;
suite.kalman_particles = 180;
suite.pit_datasets = 18;
suite.pit_trials = 70;
suite.pit_particles = 100;
suite.original_trials = 60;
suite.original_particles = 100;
suite.convergence_particles = [25 50 100];
suite.convergence_trials = 45;
suite.convergence_datasets = 4;
suite.context_trials = 100;
suite.context_particles = 120;
suite.stress_trials = 75;
suite.stress_particles = 80;
suite.benchmark_trials = 80;
suite.benchmark_particles = [50 100];
end
