function results = validate_p_values_extended(varargin)
%VALIDATE_P_VALUES_EXTENDED Posterior predictive calibration diagnostics.
%
%   For a calibrated continuous predictive CDF F_t, the PIT value
%
%       p_t = F_t(x_t)
%
%   is Uniform(0,1) over repeated draws from the data-generating process.
%   Cues are discrete, so this validator uses the randomized discrete PIT
%   F(q_t-) + U Pr(q_t), which removes the atom-induced non-uniformity of a
%   plain discrete CDF.  State and parameter ranks are included as
%   posterior diagnostics; the feedback and cue PITs are the primary
%   prequential calibration checks.

ip = inputParser;
addParameter(ip, 'NumDatasets', 25);
addParameter(ip, 'Trials', 80);
addParameter(ip, 'Particles', 100);
addParameter(ip, 'Seed', 1201);
addParameter(ip, 'MakePlots', false);
addParameter(ip, 'Strict', false);
parse(ip, varargin{:});
cfg = ip.Results;

rng(cfg.Seed);

a = 0.9;
d = 0.02;
sigmaQ = 0.03;
sigmaR = 0.05;
cueProb = [0.85 0.15; 0.15 0.85];
transition = [0.96 0.04; 0.04 0.96];
contextOffset = [-0.25 0.25];

n = cfg.NumDatasets * cfg.Trials;
feedbackP = zeros(1, n);
cueP = zeros(1, n);
stateRank = zeros(1, n);
retentionRank = zeros(1, n);
driftRank = zeros(1, n);
cursor = 0;

for dataset = 1:cfg.NumDatasets
    coin = RealTimeCOIN('num_particles', cfg.Particles, 'max_contexts', 4, ...
        'prior_mean_retention', a, 'prior_precision_retention', 1e6, ...
        'prior_mean_drift', d, 'prior_precision_drift', 1e6, ...
        'sigma_process_noise', sigmaQ, 'sigma_sensory_noise', sigmaR, ...
        'sigma_motor_noise', 0);
    c = 1;
    s = contextOffset(c);

    for t = 1:cfg.Trials
        c = validation_sample_categorical(transition(c, :));
        q = validation_sample_categorical(cueProb(c, :));
        s = a * s + d + sigmaQ * randn;
        y = s + sigmaR * randn;

        cursor = cursor + 1;
        feedbackP(cursor) = coin.predictive_state_feedback_cdf(y, q);
        cueP(cursor) = coin.predictive_cue_p_value(q);

        coin.observe_q(q);
        coin.observe_y(y);

        diag = coin.diagnostics();
        stateRank(cursor) = validation_mixture_cdf(s, diag.raw.responsibilities, ...
            diag.raw.state_filtered_mean, diag.raw.state_filtered_var);

        w = diag.responsibilities;
        w = w ./ max(sum(w, 'all'), eps);
        retentionRank(cursor) = sum(w .* (diag.retention <= a), 'all');
        driftRank(cursor) = sum(w .* (diag.drift <= d), 'all');
    end
end

feedbackP = feedbackP(1:cursor);
cueP = cueP(1:cursor);
stateRank = stateRank(1:cursor);
retentionRank = retentionRank(1:cursor);
driftRank = driftRank(1:cursor);

thresholds = struct();
thresholds.feedback_ks = 0.08;
thresholds.cue_ks = 0.08;
thresholds.state_rank_ks = 0.15;

results = struct();
results.feedback_p_values = feedbackP;
results.cue_p_values = cueP;
results.state_rank_values = stateRank;
results.retention_rank_values = retentionRank;
results.drift_rank_values = driftRank;
results.feedback_ks = validation_uniform_ks(feedbackP);
results.cue_ks = validation_uniform_ks(cueP);
results.state_rank_ks = validation_uniform_ks(stateRank);
results.retention_rank_mean = mean(retentionRank, 'omitnan');
results.drift_rank_mean = mean(driftRank, 'omitnan');
results.feedback_mean = mean(feedbackP, 'omitnan');
results.cue_mean = mean(cueP, 'omitnan');
results.thresholds = thresholds;

checks = struct();
checks.feedback_ks = results.feedback_ks < thresholds.feedback_ks;
checks.cue_ks = results.cue_ks < thresholds.cue_ks;
checks.state_rank_ks = results.state_rank_ks < thresholds.state_rank_ks;
[results.passed, results.checks] = validation_pass_summary(checks);
results.config = cfg;

fprintf('Extended PIT: feedback KS %.3f, cue KS %.3f, state-rank KS %.3f\n', ...
    results.feedback_ks, results.cue_ks, results.state_rank_ks);

if cfg.MakePlots
    figure('Name', 'RealTimeCOIN extended PIT validation');
    tiledlayout(2, 2);
    nexttile; histogram(feedbackP, 20, 'Normalization', 'probability'); title('Feedback PIT');
    nexttile; histogram(cueP, 20, 'Normalization', 'probability'); title('Cue PIT');
    nexttile; histogram(stateRank, 20, 'Normalization', 'probability'); title('State posterior rank');
    nexttile; histogram(retentionRank, 20, 'Normalization', 'probability'); title('Retention rank');
end

if cfg.Strict && ~results.passed
    error('validate_p_values_extended:Failed', 'Extended p-value validation failed.');
end
end
