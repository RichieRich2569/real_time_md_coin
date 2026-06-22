function results = validate_original_coin_monte_carlo(varargin)
%VALIDATE_ORIGINAL_COIN_MONTE_CARLO Multi-seed comparison with COIN.m.
%
%   This validator answers a different question from the Kalman test.  It
%   does not prove the model is mathematically correct from first
%   principles; instead, it checks that the streaming RealTimeCOIN
%   implementation gives the same predictive motor outputs as the original
%   off-line COIN implementation over many random seeds.

ip = inputParser;
addParameter(ip, 'Seeds', 2001:2005);
addParameter(ip, 'Trials', 60);
addParameter(ip, 'Particles', 100);
addParameter(ip, 'Strict', false);
parse(ip, varargin{:});
cfg = ip.Results;

seeds = cfg.Seeds(:)';
rmse = zeros(size(seeds));
corrValue = zeros(size(seeds));

for i = 1:numel(seeds)
    one = compare_original_coin('Trials', cfg.Trials, ...
        'Particles', cfg.Particles, 'Seed', seeds(i));
    rmse(i) = one.rmse_motor_output;
    corrValue(i) = one.correlation_motor_output;
end

thresholds = struct();
thresholds.mean_rmse = 0.03;
thresholds.worst_correlation = 0.95;

results = struct();
results.seeds = seeds;
results.rmse = rmse;
results.correlation = corrValue;
results.mean_rmse = mean(rmse, 'omitnan');
results.median_rmse = median(rmse, 'omitnan');
results.percentile95_rmse = local_percentile(rmse, 95);
results.mean_correlation = mean(corrValue, 'omitnan');
results.worst_correlation = min(corrValue);
results.rmse_motor_output = results.mean_rmse;
results.correlation_motor_output = results.mean_correlation;
results.thresholds = thresholds;

checks = struct();
checks.mean_rmse = results.mean_rmse < thresholds.mean_rmse;
checks.worst_correlation = results.worst_correlation > thresholds.worst_correlation;
[results.passed, results.checks] = validation_pass_summary(checks);
results.config = cfg;

fprintf('Original COIN Monte Carlo: mean RMSE %.4f, 95th RMSE %.4f, worst corr %.3f\n', ...
    results.mean_rmse, results.percentile95_rmse, results.worst_correlation);

if cfg.Strict && ~results.passed
    error('validate_original_coin_monte_carlo:Failed', ...
        'Original COIN Monte Carlo validation failed.');
end
end

function q = local_percentile(x, p)
x = sort(x(isfinite(x)));
if isempty(x)
    q = NaN;
    return;
end
idx = 1 + (numel(x) - 1) * p / 100;
lo = floor(idx);
hi = ceil(idx);
if lo == hi
    q = x(lo);
else
    q = x(lo) + (idx - lo) * (x(hi) - x(lo));
end
end
