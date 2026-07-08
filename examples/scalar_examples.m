%[text] # RealTimeCOIN - scalar examples
%[text] Companion notebook: `md_examples.m` (the multi-dimensional cases).
%[text] ## Section 1 - Setup
%[text] Put the RealTimeCOIN class (repo root) and the coinviz package (this folder) on the path. Run this section once before the others.
% Locate the repo root by looking for its folders (@RealTimeCOIN and
% examples/+coinviz). We can't trust mfilename here: pressing "Run Section"
% in the MATLAB Editor executes a copy of the code from a temp folder, so
% mfilename('fullpath') points at that temp path, not this file.
here = fileparts(mfilename('fullpath'));
repoRoot = '';
for c = {here, fileparts(here), pwd, fileparts(pwd)}
    d = c{1};
    if ~isempty(d) && isfolder(fullfile(d, '@RealTimeCOIN')) ...
            && isfolder(fullfile(d, 'examples', '+coinviz'))
        repoRoot = d; break;
    end
end
if isempty(repoRoot)
    error(['Cannot locate the RealTimeCOIN repo. cd to the repo root ' ...
        '(the folder containing @RealTimeCOIN and examples/) and re-run this section.']);
end
addpath(repoRoot);                                % RealTimeCOIN class
addpath(fullfile(repoRoot, 'examples'));          % +coinviz package
fprintf('Paths added. RealTimeCOIN on path: %d\n', exist('RealTimeCOIN', 'class') == 8);
%%
%[text] ## Section 2 - Bias inference on a short cue sequence
%[text] The smallest demo: a handful of cued trials with noisy scalar feedback. The model infers, per trial, a posterior over the latent state (via numerical integration of state\_probability) and a distribution over contexts.
rng(1);
coin = RealTimeCOIN('num_particles', 200, 'max_contexts', 5, 'infer_bias', true);

cues        = [1 1 2 2 1 3 1];
trueValues  = [0.2 0.2 0.5 0.5 0.2 -0.1 0.2];        % latent target per trial
feedbacks   = trueValues + coin.sigma_sensory_noise * randn(size(trueValues));

grid = linspace(-1.5, 1.5, 201);
T = numel(cues);
predMean = zeros(1, T);
for t = 1:T
    coin.observe_q(cues(t));
    coin.observe_y(feedbacks(t));

    dens = coin.state_probability(grid);             % posterior state density on the grid
    a = trapz(grid, dens); if a > 0, dens = dens / a; end
    predMean(t) = trapz(grid, dens .* grid);         % posterior mean by integration

    probs = coin.responsibilities_map();         % containers.Map: context -> responsibility
    fprintf('Trial %d (cue %d): E[state]=% .3f, contexts={', t, cues(t), predMean(t));
    kk = cell2mat(probs.keys);
    for k = kk, fprintf(' %d:%.2f', k, probs(k)); end
    fprintf(' }\n');
end
fprintf('Final trial count (Trial property): %d\n', coin.Trial);

coinviz.stateTrace(1:T, feedbacks, trueValues, predMean, [], ...
    'FigName', 'Scalar: short cue sequence', 'PredictedName', 'Posterior mean');
%%
%[text] ## Section 3 - Long run with missing observations
%[text] A 340-trial perturbation schedule that mirrors the original COIN tests: a baseline, a positive block, a short negative block, then a long stretch of **missing** feedback (NaN), which the model must coast through on its dynamics prior alone. The final figure shows the per-context posterior state density.
rng(3);
coin = RealTimeCOIN();                               % defaults (scalar)

truePert = [zeros(1,50), ones(1,125), -ones(1,15), NaN(1,150)];
obs = truePert + coin.sigma_sensory_noise * randn(size(truePert));

T = numel(obs);
ctxWidth = coin.max_contexts + 1;
predMean = zeros(1, T);
postCtx  = zeros(T, ctxWidth);
grid = linspace(-1.5, 1.5, 201);
for t = 1:T
    coin.observe_y(obs(t));                     % NaN = missing observation
    p = coin.context_responsibilities_local();        % fast local responsibilities
    postCtx(t, 1:numel(p)) = p;
    dens = coin.state_probability(grid);
    a = trapz(grid, dens); if a > 0, dens = dens / a; end
    predMean(t) = trapz(grid, dens .* grid);
end

coinviz.stateTrace(1:T, obs, truePert, predMean, [], ...
    'FigName', 'Scalar: long run (missing obs)', 'PredictedName', 'Posterior mean');
coinviz.contextBars([], postCtx, 'FigName', 'Scalar: context responsibilities', ...
    'PostTitle', 'Local/modal context responsibilities');

stateByCtx = coin.state_given_context_probability(grid);   % Map: context -> density
coinviz.densityLines(grid, stateByCtx, 'FigName', 'Scalar: per-context state density', ...
    'Title', 'Final posterior state density given aligned context', 'XLabel', 'State value');
%%
%[text] ## Section 4 - Tuned "coin-rl" settings with cues
%[text] Same idea as Section 3 but with custom noise/retention priors (the settings used by the coin-rl tests) and an explicit cue stream, so we can compare the **predicted** context probabilities (before each observation) against the **responsibilities** (after).
rng(3);
priorMeanRetention = 0.9425;
coin = RealTimeCOIN('sigma_sensory_noise', 0.03, 'sigma_motor_noise', 0.0182, ...
    'prior_mean_retention', priorMeanRetention, ...
    'prior_precision_drift', 1 / (0.0001 + (1 - priorMeanRetention^2)));

blockPert = [0, 0.5, -0.5, 0.5];         % target per block
blockCues = [1, 2, 3, 2];                % cue per block
blockLen  = 80;
truePert = repelem(blockPert, blockLen);
cues     = repelem(blockCues, blockLen);
obs = truePert + coin.sigma_sensory_noise * randn(size(truePert));

T = numel(obs);
ctxWidth = coin.max_contexts + 1;
predMean = zeros(1, T);
prevCtx  = zeros(T, ctxWidth);
postCtx  = zeros(T, ctxWidth);
grid = linspace(-1.5, 1.5, 201);
for t = 1:T
    coin.observe_q(cues(t));
    pv = coin.predicted_context_probabilities_local();
    prevCtx(t, 1:numel(pv)) = pv;

    coin.observe_y(obs(t));
    rv = coin.context_responsibilities_local();
    postCtx(t, 1:numel(rv)) = rv;

    dens = coin.state_probability(grid);
    a = trapz(grid, dens); if a > 0, dens = dens / a; end
    predMean(t) = trapz(grid, dens .* grid);
end
blockEdges = blockLen * (1:numel(blockCues)-1);

coinviz.stateTrace(1:T, obs, truePert, predMean, blockEdges, ...
    'FigName', 'Scalar: tuned settings', 'PredictedName', 'Posterior mean');
coinviz.contextBars(prevCtx, postCtx, 'FigName', 'Scalar: predicted vs responsibilities');
coinviz.densityLines(grid, coin.state_given_context_probability(grid), ...
    'FigName', 'Scalar: per-context state density (tuned)', ...
    'Title', 'Per-context state density', 'XLabel', 'State value');
%%
%[text] ## Section 5 - Save, load and stationarize
%[text] saveModel serializes a trained model to a .mat file; loadModel restores it into a fresh object. With setStationary = true the saved model is first put into its steady state (Trial reset to 0, context probabilities set to the stationary distribution) so it can be reused as a prior. set\_stationary does that in place. Here we verify a load round-trip reproduces the responsibilities.
rng(4);
coin = RealTimeCOIN('infer_bias', true);
cueSeq = [1 1 2 2 1 2 1 2];
valSeq = [0.2 0.2 -0.3 -0.3 0.2 -0.3 0.2 -0.3];
for t = 1:numel(cueSeq)
    coin.observe_q(cueSeq(t));
    coin.observe_y(valSeq(t) + coin.sigma_sensory_noise * randn);
end
fprintf('Before save: Trial = %d, active contexts = %d\n', coin.Trial, coin.responsibilities_map().Count);

tmpFile = [tempname '.mat'];

% (a) Non-stationary save preserves the live trial state exactly.
coin.saveModel(tmpFile, false);
reloaded = RealTimeCOIN('infer_bias', true);
reloaded.loadModel(tmpFile);
r1 = coin.responsibilities_map();
r2 = reloaded.responsibilities_map();
maxDiff = max(abs(cell2mat(r1.values) - cell2mat(r2.values)));
fprintf('Non-stationary reload: Trial = %d, max responsibility diff = %.2e\n', ...
    reloaded.Trial, maxDiff);

% (b) Stationary save resets Trial while retaining learned contexts.
coin.saveModel(tmpFile, true);
stationaryModel = RealTimeCOIN('infer_bias', true);
stationaryModel.loadModel(tmpFile);
fprintf('Stationary reload:     Trial = %d (reset), active contexts = %d (retained)\n', ...
    stationaryModel.Trial, stationaryModel.responsibilities_map().Count);

% (c) set_stationary applied in place to a copy trained the same way. It drives
%     the model onto the stationary distribution of its own learned context
%     transition matrix, so the analytic stationary_context_probabilities
%     computed *before* the reset matches the distribution the model actually
%     adopts *after* it (over the instantiated contexts).
piAnalytic = coin.stationary_context_probabilities();   % 1-by-K analytic stationary
coin.set_stationary();
fprintf('After set_stationary:  Trial = %d\n', coin.Trial);

predicted = coin.predicted_context_probabilities_vector();     % 1-by-(max_contexts+1)
resp      = coin.responsibilities_vector();
Kk        = numel(piAnalytic);
predKnown = predicted(1:Kk) / sum(predicted(1:Kk));     % drop the novel slot, renormalise
fprintf('stationary_context_probabilities (analytic) : [%s]\n', num2str(piAnalytic, '%.3f '));
fprintf('predicted context probs after (known/renorm): [%s]\n', num2str(predKnown, '%.3f '));
fprintf('max|analytic - adopted| = %.3f\n', max(abs(piAnalytic - predKnown)));
% After the reset the chain sits at its fixed point, so predicted == responsibilities.
assert(max(abs(predicted(:) - resp(:))) < 1e-9, ...
    'predicted == responsibilities after set_stationary');
assert(max(abs(piAnalytic - predKnown)) < 0.1, ...
    'model settles onto its analytic stationary distribution');
delete(tmpFile);
%%
%[text] ## Section 6 - Method coverage (remaining scalar API)
%[text] This section deliberately calls every public method not already exercised above, so the notebook demonstrates the full scalar API surface. It trains a short model and then queries it every which way, printing a checklist.
rng(5);
coin = RealTimeCOIN('infer_bias', true);
cueSeq = [1 1 2 2 1 3 1 2 3 1];
valSeq = [0.2 0.2 -0.3 -0.3 0.2 0.6 0.2 -0.3 0.6 0.2];
for t = 1:numel(cueSeq)
    coin.observe_q(cueSeq(t));
    coin.observe_y(valSeq(t) + coin.sigma_sensory_noise * randn);
end
grid = linspace(-1.5, 1.5, 201);

fprintf('\n================ Scalar method coverage ================\n');
[mS, vS] = coin.state_moments();
fprintf('state_moments ................ mean % .3f, var %.4f\n', mS, vS);
fprintf('motor_output ................. % .3f\n', coin.motor_output());
fprintf('predictive_motor_output(1) ... % .3f\n', coin.predictive_motor_output(1));
[mF, vF] = coin.predictive_feedback_moments(1);
fprintf('predictive_feedback_moments(1) mean % .3f, var %.4f\n', mF, vF);
fprintf('predictive_state_feedback_cdf(0.2) . %.3f\n', coin.predictive_state_feedback_cdf(0.2, 1));
fprintf('predictive_cue_p_value(1, 0.5) ..... %.3f\n', coin.predictive_cue_p_value(1, 0.5));

% Grid densities (feedback space and per-context feedback space).
fbDens = coin.state_feedback_probability(grid);
fprintf('state_feedback_probability ... integral %.3f\n', trapz(grid, fbDens));
coinviz.densityLines(grid, coin.state_feedback_given_context_probability(grid), ...
    'FigName', 'Scalar: per-context feedback density', ...
    'Title', 'state\_feedback\_given\_context\_probability', 'XLabel', 'Feedback', ...
    'NovelDensity', coin.novel_state_feedback_probability(grid));

% Novel (not-yet-instantiated) context densities overlaid on the state density.
coinviz.densityLines(grid, coin.state_given_context_probability(grid), ...
    'FigName', 'Scalar: state density + novel context', ...
    'Title', 'state\_given\_context\_probability with novel overlay', 'XLabel', 'State value', ...
    'NovelDensity', coin.novel_state_probability(grid));

% Context summaries: global Maps, global vectors, and the alignment struct.
cpm = coin.predicted_context_probabilities_map();     % Map
crm = coin.responsibilities_map();            % Map
fprintf('context_predicted_probabilities .. %d contexts\n', cpm.Count);
fprintf('context_responsibilities ......... %d contexts\n', crm.Count);
fprintf('predicted_context_probabilities .. [%s]\n', num2str(coin.predicted_context_probabilities_vector(), '%.2f '));
fprintf('responsibilities ................. [%s]\n', num2str(coin.responsibilities_vector(), '%.2f '));
fprintf('sampled_context_count ............ [%s]\n', num2str(coin.sampled_context_count(), '%.2f '));
fprintf('sampled_context_count_local ...... [%s]\n', num2str(coin.sampled_context_count_local(), '%.2f '));
al = coin.context_alignment();
fprintf('context_alignment ................ struct with fields: %s\n', strjoin(fieldnames(al)', ', '));

% Per-trial c*/component scalars (the COIN "single most likely context" traces).
fprintf('explicit_component ........... % .3f\n', coin.explicit_component());
fprintf('implicit_component ........... % .3f\n', coin.implicit_component());
fprintf('state_cstar1/2/3 ............. % .3f / % .3f / % .3f\n', ...
    coin.state_cstar1(), coin.state_cstar2(), coin.state_cstar3());
fprintf('predicted_probability_cstar1/3 %.3f / %.3f\n', ...
    coin.predicted_probability_cstar1(), coin.predicted_probability_cstar3());
fprintf('kalman_gain_cstar1/2 ......... %.3f / %.3f\n', ...
    coin.kalman_gain_cstar1(), coin.kalman_gain_cstar2());

% Transition / cue / stationary distributions over contexts.
ltp = coin.local_transition_probabilities();
lcp = coin.local_cue_probabilities();
scp = coin.stationary_context_probabilities();
gtp = coin.global_transition_probabilities();
gcp = coin.global_cue_probabilities();
fprintf('local_transition_probabilities  %d-by-%d (rows sum to 1: %d)\n', ...
    size(ltp, 1), size(ltp, 2), all(abs(sum(ltp, 2) - 1) < 1e-9));
fprintf('local_cue_probabilities ...... %d-by-%d (rows sum to 1: %d)\n', ...
    size(lcp, 1), size(lcp, 2), all(abs(sum(lcp, 2) - 1) < 1e-9));
fprintf('stationary_context_probabilities [%s] (sum %.3f)\n', num2str(scp, '%.3f '), sum(scp));
fprintf('global_transition_probabilities  [%s] (sum %.3f)\n', num2str(gtp, '%.2f '), sum(gtp));
fprintf('global_cue_probabilities ..... [%s] (sum %.3f)\n', num2str(gcp, '%.2f '), sum(gcp));
assert(all(abs(sum(ltp, 2) - 1) < 1e-9), 'local transition rows must sum to 1');
assert(abs(sum(scp) - 1) < 1e-9, 'stationary distribution must sum to 1');

fprintf('=======================================================\n');

% Per-context parameter densities (scalar-dynamics only): retention, drift, bias.
rGrid = linspace(0.80, 1.00, 201);
dGrid = linspace(-0.06, 0.06, 201);
bGrid = linspace(-0.80, 0.80, 201);
coinviz.densityLines(rGrid, coin.retention_given_context_probability(rGrid), ...
    'FigName', 'Scalar: retention | context', ...
    'Title', 'retention\_given\_context\_probability', 'XLabel', 'Retention a');
coinviz.densityLines(dGrid, coin.drift_given_context_probability(dGrid), ...
    'FigName', 'Scalar: drift | context', ...
    'Title', 'drift\_given\_context\_probability', 'XLabel', 'Drift d');
coinviz.densityLines(bGrid, coin.bias_given_context_probability(bGrid), ...
    'FigName', 'Scalar: bias | context', ...
    'Title', 'bias\_given\_context\_probability', 'XLabel', 'Bias');
% Marginal (across-context) bias density.
figure('Name', 'Scalar: marginal bias density');
plot(bGrid, coin.bias_probability(bGrid), 'LineWidth', 1.4);
xlabel('Bias'); ylabel('Density'); title('bias\_probability (marginal)');
%%
%[text] ## Section 7 - Explicit vs implicit decomposition (c\* traces)
%[text] The COIN adaptation curve can be read out in several ways. explicit\_component is the state of the single most-responsible context (== state\_cstar1); implicit\_component is the motor output minus the average state; and the c\* traces track the most-probable context's state under different timing conventions. Here we record them through a perturbation that flips sign.
rng(7);
coin = RealTimeCOIN('infer_bias', true);
perturb = [zeros(1, 10), 0.3 * ones(1, 20), -0.3 * ones(1, 20), zeros(1, 10)];
T = numel(perturb);
mo = zeros(1, T); ex = zeros(1, T); im = zeros(1, T);
cs1 = zeros(1, T); cs2 = zeros(1, T); cs3 = zeros(1, T);
for t = 1:T
    coin.observe_y(perturb(t) + coin.sigma_sensory_noise * randn);
    mo(t)  = coin.motor_output();
    ex(t)  = coin.explicit_component();
    im(t)  = coin.implicit_component();
    cs1(t) = coin.state_cstar1();
    cs2(t) = coin.state_cstar2();
    cs3(t) = coin.state_cstar3();
end
blockEdges = [10 30 50];
cols = coinviz.palette(4);
figure('Name', 'Scalar: explicit / implicit read-outs');
hold on;
plot(1:T, perturb, 'k--', 'LineWidth', 1.2);
plot(1:T, mo, 'Color', cols(1, :), 'LineWidth', 1.6);
plot(1:T, ex, 'Color', cols(2, :), 'LineWidth', 1.4);
plot(1:T, im, 'Color', cols(3, :), 'LineWidth', 1.4);
for e = blockEdges, xline(e, 'Color', [0.5 0.5 0.5 0.4]); end
xlabel('Trial'); ylabel('Adaptation');
title('motor\_output, explicit\_component and implicit\_component');
legend({'perturbation', 'motor\_output', 'explicit', 'implicit'}, 'Location', 'best');
% The c* state estimates all track the dominant context's state.
figure('Name', 'Scalar: c* state estimates');
hold on;
plot(1:T, perturb, 'k--', 'LineWidth', 1.2);
plot(1:T, cs1, 'Color', cols(1, :), 'LineWidth', 1.4);
plot(1:T, cs2, 'Color', cols(2, :), 'LineWidth', 1.4);
plot(1:T, cs3, 'Color', cols(3, :), 'LineWidth', 1.4);
for e = blockEdges, xline(e, 'Color', [0.5 0.5 0.5 0.4]); end
xlabel('Trial'); ylabel('State estimate');
title('state\_cstar1 / state\_cstar2 / state\_cstar3');
legend({'perturbation', 'cstar1', 'cstar2', 'cstar3'}, 'Location', 'best');
% explicit_component is exactly state_cstar1 by construction.
assert(max(abs(ex - cs1)) < 1e-9, 'explicit_component == state_cstar1');
fprintf('Section 7: max|explicit - state_cstar1| = %.2e\n', max(abs(ex - cs1)));
%%
%[text] ## Section 8 - Prior / hyperparameter exploration
%[text] The same feedback stream shown to two differently-tuned models: an "eager" prior (small alpha\_context, large rho\_context) that instantiates new contexts readily, versus a "sticky" prior that resists switching. We compare context creation and the c\* state read-out. The last (grey) bar is the novel context.
rng(8);
perturb = [zeros(1, 8), 0.4 * ones(1, 16), -0.4 * ones(1, 16)];
T = numel(perturb);
fb = perturb + 0.03 * randn(1, T);          % identical feedback both models see

eager  = RealTimeCOIN('alpha_context', 2,  'rho_context', 0.6);
sticky = RealTimeCOIN('alpha_context', 30, 'rho_context', 0.05);
prevE = zeros(T, eager.max_contexts + 1);
prevS = zeros(T, sticky.max_contexts + 1);
csE = zeros(1, T); csS = zeros(1, T);
for t = 1:T
    eager.observe_y(fb(t));
    sticky.observe_y(fb(t));
    prevE(t, :) = eager.predicted_context_probabilities_vector();
    prevS(t, :) = sticky.predicted_context_probabilities_vector();
    csE(t) = eager.state_cstar1();
    csS(t) = sticky.state_cstar1();
end
coinviz.contextBars(prevE, [], 'FigName', 'Eager prior: context creation', ...
    'NovelContext', true, 'PrevTitle', 'Eager: predicted context probabilities');
coinviz.contextBars(prevS, [], 'FigName', 'Sticky prior: context creation', ...
    'NovelContext', true, 'PrevTitle', 'Sticky: predicted context probabilities');
figure('Name', 'Prior comparison: c*1 state');
hold on;
plot(1:T, fb, 'x', 'Color', [0.7 0.7 0.7], 'MarkerSize', 4);
plot(1:T, csE, 'Color', cols(1, :), 'LineWidth', 1.5);
plot(1:T, csS, 'Color', cols(2, :), 'LineWidth', 1.5);
xlabel('Trial'); ylabel('state\_cstar1');
legend({'feedback', 'eager', 'sticky'}, 'Location', 'best');
title('Prior effect on the dominant-context state');
fprintf('Eager model contexts: %d;  Sticky model contexts: %d\n', ...
    eager.diagnostics().C, sticky.diagnostics().C);
%%
%[text] ## Section 9 - State|context probability evolution (composite heat-map)
%[text] Each trial we evaluate state\_given\_context\_probability on a fixed grid and stack the columns into one image. Every context is tinted its own colour and the densities are additively blended, so you can watch contexts claim regions of state space as the contingency changes; the novel context is grey. Below the figure we print every inferred per-context parameter.
rng(9);
coin = RealTimeCOIN('infer_bias', true);
perturb = [0.20 * ones(1, 15), -0.35 * ones(1, 15), 0.20 * ones(1, 15)];
cues    = [ones(1, 15), 2 * ones(1, 15), ones(1, 15)];
T = numel(perturb);
sGrid = linspace(-0.8, 0.8, 161);
recorded = containers.Map('KeyType', 'double', 'ValueType', 'any');
novelDens = zeros(numel(sGrid), T);
trueLine = zeros(1, T);
for t = 1:T
    coin.observe_q(cues(t));
    coin.observe_y(perturb(t) + coin.sigma_sensory_noise * randn);
    dmap = coin.state_given_context_probability(sGrid);
    for kk = cell2mat(dmap.keys)
        if ~isKey(recorded, kk)
            recorded(kk) = zeros(numel(sGrid), T);
        end
        col = recorded(kk); col(:, t) = dmap(kk)'; recorded(kk) = col;
    end
    novelDens(:, t) = coin.novel_state_probability(sGrid)';
    trueLine(t) = perturb(t);
end
coinviz.contextDensityEvolution(1:T, sGrid, recorded, ...
    'FigName', 'Scalar: state|context evolution', ...
    'Title', 'state\_given\_context\_probability over trials (colours = contexts)', ...
    'NovelDens', novelDens, 'TrueLine', trueLine, 'BlockEdges', [15 30]);
% Diagnostics: every inferred per-context parameter.
D = coin.diagnostics();
fprintf('\nInferred contexts (K = %d):\n', D.C);
for c = 1:D.C
    fprintf('  ctx %d: retention %.3f  drift % .4f  bias % .3f  state_mean % .3f\n', ...
        c, D.retention(c), D.drift(c), D.bias(c), D.state_mean(c));
end
%%
%[text] ## Section 10 - Parallel runs: probability averaging across an ensemble
%[text] `RealTimeCOINEnsemble` runs R independent `RealTimeCOIN` filters on the **same** feedback stream and returns their equal-weight average. Each run draws from its own reproducible RNG substream (seeded from `seed`), so the runs are independent yet the whole ensemble is deterministic given `seed`. A single run is a noisy Monte-Carlo estimate of the model's expected output; averaging R runs cuts that estimation noise by roughly a factor of sqrt(R). Below, six individual runs (thin) scatter around the 30-run ensemble average (bold).
rng(10);
perturb = [zeros(1, 10), 0.4 * ones(1, 25), -0.3 * ones(1, 25), zeros(1, 15)];
T = numel(perturb);
fb = perturb + 0.03 * randn(1, T);                    % one feedback stream, shared by all runs
nShow = 6; np = 40;

indiv = zeros(nShow, T);                               % individual single-run trajectories
for s = 1:nShow
    m = RealTimeCOINEnsemble('runs', 1, 'seed', s, 'num_particles', np);
    for t = 1:T, m.observe_y(fb(t)); indiv(s, t) = m.motor_output(); end
end
ens = RealTimeCOINEnsemble('runs', 30, 'seed', 101, 'num_particles', np);
moEns = zeros(1, T);
for t = 1:T, ens.observe_y(fb(t)); moEns(t) = ens.motor_output(); end

cols = coinviz.palette(3);
figure('Name', 'Ensemble: probability averaging across runs');
hold on;
hP = plot(1:T, perturb, 'k--', 'LineWidth', 1.2);
hI = plot(1:T, indiv', '-', 'Color', [cols(2, :), 0.35], 'LineWidth', 0.8);
hE = plot(1:T, moEns, '-', 'Color', cols(1, :), 'LineWidth', 2.0);
xlabel('Trial'); ylabel('motor\_output');
legend([hP, hI(1), hE], {'perturbation', 'individual runs', '30-run ensemble average'}, ...
    'Location', 'best');
title('Six single runs (thin) vs the 30-run ensemble average (bold)');
acrossRunStd = mean(std(indiv, 0, 1));
fprintf('Mean across-run std of a single run = %.4f;  ~30-run estimator SE = %.4f (that / sqrt(30))\n', ...
    acrossRunStd, acrossRunStd / sqrt(30));

% Reproducibility: same seed => bit-identical ensemble.
ensA = RealTimeCOINEnsemble('runs', 8, 'seed', 99, 'num_particles', 80);
ensB = RealTimeCOINEnsemble('runs', 8, 'seed', 99, 'num_particles', 80);
for t = 1:T, ensA.observe_y(fb(t)); ensB.observe_y(fb(t)); end
fprintf('Reproducibility (same seed): max|motor diff| = %.2e;  weights sum = %.3f;  Trial = %d\n', ...
    abs(ensA.motor_output() - ensB.motor_output()), sum(ens.weights), ens.Trial);
%%
%[text] ## Section 11 - Batch replay across runs with `parfor`
%[text] When the whole observation sequence is known in advance, `simulate(qSeq, ySeq)` replays every run in a single call and returns per-trial run-averaged traces: `motor_output`, the pooled `state_mean`, and the pooled predictive `state_var`. With `max_cores > 0` the runs are dispatched across workers with `parfor`; the result is **bit-identical** to the serial path (and to trial-by-trial stepping). The shaded band is +/- one pooled predictive standard deviation.
rng(11);
perturb = [zeros(1, 20), 0.5 * ones(1, 40), -0.4 * ones(1, 40), zeros(1, 20)];
cues = ones(1, numel(perturb));
obs  = perturb + 0.03 * randn(size(perturb));
T = numel(perturb);

ensSerial = RealTimeCOINEnsemble('runs', 50, 'seed', 7, 'max_cores', 0, 'num_particles', 100);
ensPar    = RealTimeCOINEnsemble('runs', 50, 'seed', 7, 'max_cores', 8, 'num_particles', 100);
tic; trS = ensSerial.simulate(cues, obs); tSer = toc;
tic; trP = ensPar.simulate(cues, obs);    tPar = toc;   % parfor across runs (first call also starts the pool)
fprintf('simulate 50 runs: serial %.2fs, parallel %.2fs\n', tSer, tPar);
fprintf('serial vs parallel: max|motor diff| = %.2e (bit-identical)\n', ...
    max(abs(trS.motor_output - trP.motor_output)));

mu = trP.motor_output;
sd = sqrt(max(trP.state_var, 0));
cols = coinviz.palette(3);
figure('Name', 'Ensemble: batch simulate across runs (parfor)');
hold on;
fill([1:T, T:-1:1], [mu + sd, fliplr(mu - sd)], cols(1, :), 'FaceAlpha', 0.15, 'EdgeColor', 'none');
plot(1:T, perturb, 'k--', 'LineWidth', 1.2);
plot(1:T, mu, '-', 'Color', cols(1, :), 'LineWidth', 1.8);
xlabel('Trial'); ylabel('motor\_output');
legend({'\pm 1 pooled SD', 'perturbation', 'run-averaged motor\_output'}, 'Location', 'best');
title('Batch replay of 50 runs with parfor');
%%
%[text] ## Section 12 - Ensemble context-aligned summaries
%[text] Context labels are arbitrary in each run, so before averaging any context-indexed quantity the ensemble matches every run's contexts onto a common reference frame (by prototype similarity). The averaged context-probability vectors still sum to 1, and each per-context state density is averaged over the runs that instantiated that context. An A/B/A cued schedule instantiates two contexts across 30 runs.
rng(12);
blockPert = [0.3, -0.3, 0.3];
blockCue  = [1, 2, 1];
blockLen  = 25;
perturb = repelem(blockPert, blockLen);
cues    = repelem(blockCue, blockLen);
obs     = perturb + 0.03 * randn(size(perturb));
T = numel(perturb);
maxCtx = 5;

ens = RealTimeCOINEnsemble('runs', 30, 'seed', 5, 'max_contexts', maxCtx, 'num_particles', 100);
prevCtx = zeros(T, maxCtx + 1);
postCtx = zeros(T, maxCtx + 1);
for t = 1:T
    ens.observe_q(cues(t));
    prevCtx(t, :) = ens.predicted_context_probabilities_vector();   % run-averaged, aligned
    ens.observe_y(obs(t));
    postCtx(t, :) = ens.responsibilities_vector();
end
coinviz.contextBars(prevCtx, postCtx, 'FigName', 'Ensemble: aligned context probabilities', ...
    'NovelContext', true, 'PrevTitle', 'Run-averaged predicted (aligned)', ...
    'PostTitle', 'Run-averaged responsibilities (aligned)');

grid = linspace(-0.8, 0.8, 201);
coinviz.densityLines(grid, ens.state_given_context_probability(grid), ...
    'FigName', 'Ensemble: per-context state density', ...
    'Title', 'Run-averaged state\_given\_context\_probability', 'XLabel', 'State value');

resp = ens.responsibilities_vector();
piE  = ens.stationary_context_probabilities();
fprintf('Run-averaged responsibilities : [%s] (sum %.3f)\n', num2str(resp, '%.2f '), sum(resp));
fprintf('Ensemble stationary context   : [%s] (sum %.3f)\n', num2str(piE, '%.3f '), sum(piE));
assert(abs(sum(resp) - 1) < 1e-9, 'aligned responsibilities sum to 1');

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":36}
