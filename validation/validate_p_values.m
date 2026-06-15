function results = validate_p_values(varargin)
%VALIDATE_P_VALUES Posterior predictive p-value validation for RealTimeCOIN.
%
%   results = validate_p_values('NumDatasets', 100, 'Trials', 200)
%   generates scalar synthetic streams, scores each cue/feedback before it
%   is consumed, and reports Kolmogorov-Smirnov style distances from a
%   uniform distribution. This is validation code, not a fast unit test.

ip = inputParser;
addParameter(ip, 'NumDatasets', 100);
addParameter(ip, 'Trials', 200);
addParameter(ip, 'Particles', 100);
addParameter(ip, 'Seed', 1001);
addParameter(ip, 'MakePlots', false);
parse(ip, varargin{:});
cfg = ip.Results;

rng(cfg.Seed);

a = 0.9;
d = 0.02;
sigmaQ = 0.03;
sigmaR = 0.05;
cueProb = [0.85 0.15; 0.15 0.85];
transition = [0.96 0.04; 0.04 0.96];
contextState = [-0.25 0.25];

feedbackP = zeros(1, cfg.NumDatasets * cfg.Trials);
cueP = zeros(1, cfg.NumDatasets * cfg.Trials);
cursor = 0;

for dataset = 1:cfg.NumDatasets
    coin = RealTimeCOIN('num_particles', cfg.Particles, 'max_contexts', 4, ...
        'prior_mean_retention', a, 'prior_precision_retention', 1e6, ...
        'prior_mean_drift', d, 'prior_precision_drift', 1e6, ...
        'sigma_process_noise', sigmaQ, 'sigma_sensory_noise', sigmaR, ...
        'sigma_motor_noise', 0);
    c = 1;
    s = contextState(c);
    for t = 1:cfg.Trials
        c = sample_categorical(transition(c,:));
        q = sample_categorical(cueProb(c,:));
        s = a * s + d + sigmaQ * randn;
        y = s + sigmaR * randn;

        cursor = cursor + 1;
        feedbackP(cursor) = coin.predictive_state_feedback_cdf(y, q);
        cueP(cursor) = coin.predictive_cue_p_value(q);

        coin.observe_q(q);
        coin.observe_y(y);
    end
end

feedbackP = feedbackP(1:cursor);
cueP = cueP(1:cursor);
results = struct();
results.feedback_p_values = feedbackP;
results.cue_p_values = cueP;
results.feedback_ks = uniform_ks(feedbackP);
results.cue_ks = uniform_ks(cueP(~isnan(cueP)));
results.feedback_mean = mean(feedbackP);
results.cue_mean = mean(cueP(~isnan(cueP)));
results.config = cfg;

fprintf('Feedback p-value KS distance: %.3f, mean: %.3f\n', results.feedback_ks, results.feedback_mean);
fprintf('Cue p-value KS distance: %.3f, mean: %.3f\n', results.cue_ks, results.cue_mean);

if cfg.MakePlots
    figure('Name', 'RealTimeCOIN p-value validation');
    tiledlayout(1,2);
    nexttile; histogram(feedbackP, 20, 'Normalization', 'probability'); title('Feedback p-values');
    nexttile; histogram(cueP, 20, 'Normalization', 'probability'); title('Cue p-values');
end
end

function idx = sample_categorical(p)
idx = find(rand <= cumsum(p ./ sum(p)), 1);
end

function d = uniform_ks(p)
p = sort(p(:));
n = numel(p);
if n == 0
    d = NaN;
    return;
end
grid = (1:n)' ./ n;
d = max(max(abs(grid - p)), max(abs(([0:n-1]' ./ n) - p)));
end
