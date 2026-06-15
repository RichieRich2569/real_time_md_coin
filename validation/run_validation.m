function results = run_validation
%RUN_VALIDATION Run compact RealTimeCOIN validation checks.
%
%   These checks are intentionally smaller than a publication-scale
%   calibration run. Increase NumDatasets/Trials in the individual
%   functions for long validation.

rootDir = fileparts(fileparts(mfilename('fullpath')));
addpath(rootDir);
addpath(fullfile(rootDir, 'validation'));

results = struct();
results.p_values = validate_p_values('NumDatasets', 25, 'Trials', 80, 'Particles', 100);
results.original_comparison = compare_original_coin('Trials', 60, 'Particles', 100);
results.performance = benchmark_realtimecoin('Trials', 80, 'Particles', [50 100]);
end
