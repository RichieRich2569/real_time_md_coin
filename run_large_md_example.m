%RUN_LARGE_MD_EXAMPLE Long-run demonstration of the multi-dimensional RealTimeCOIN.
%
%   Runs the N-dimensional model for 500 trials over a sequence of distinct
%   2-D "constant contingencies" (each a different context), added with cues and each with its own latent
%   target, including a recall block, to demonstrate multi-context adaptation.
%   It exercises and reports every major element of the implementation:
%
%     * State tracking      - posterior state mean vs. true target vs. noisy
%                             observation, per dimension and in 2-D.
%     * Context prediction  - fast local/modal summaries ("prev", before
%                             each observation).
%     * Context responsibility - fast local/modal summaries ("post", after
%                             each observation).
%     * Learned parameters  - per-context retention A, drift d, bias b, the
%                             filtered state mean/covariance and the cue
%                             association, read from diagnostics() at the end.
%
%   Companion to run_large_example.m (scalar). Multi-dimensional summaries use
%   the MD-capable query API, including the grid-based predictive densities
%   (state_probability, state_feedback_probability,
%   state_given_context_probability) and the marginal predictive CDF
%   (predictive_state_feedback_cdf), demonstrated in the final figures.

clear; clc;

% Add path to this directory if necessary
if ~exist('RealTimeCOIN', 'class')
    addpath(fileparts(mfilename('fullpath')));
end

rng(11);

N = 2;                                   % state / observation dimensionality
maxContexts = 6;
Cwidth = maxContexts + 1;                % context-vector width (incl. novel slot)

coin = RealTimeCOIN('num_particles', 200, 'max_contexts', maxContexts, ...
    'state_dim', N, 'infer_bias', true);

% -------------------------------------------------------------------------
% Build the experiment: a sequence of cued 2-D contingencies. Each context
% has a distinct latent target; the schedule includes a recall of context 1
% to show that a previously learned contingency is re-engaged rather than
% relearned from scratch.
% -------------------------------------------------------------------------
targets = [ 0.6  -0.4   0.2   0.6 ;     % dim-1 targets for the 4 schedule blocks
           -0.3   0.5   0.6  -0.3 ];    % dim-2 targets
blockCues   = [1 2 3 1];                 % cue presented in each block
blockLens   = [150 150 130 70];          % trial counts (sum = 500)

T = sum(blockLens);
cues = zeros(1, T);
trueState = zeros(N, T);
bnd = [0 cumsum(blockLens)];
for b = 1:numel(blockLens)
    idx = (bnd(b)+1):bnd(b+1);
    cues(idx) = blockCues(b);
    trueState(:, idx) = repmat(targets(:, b), 1, numel(idx));
end

% Noisy observations around the (piecewise-constant) latent target.
obsSigma = coin.sigma_sensory_noise;
feedbacks = trueState + obsSigma * 0.01 * randn(N, T);

% -------------------------------------------------------------------------
% Run the model, recording everything needed for confirmation.
% -------------------------------------------------------------------------
postStateMean  = zeros(N, T);            % posterior E[state] each trial
predFeedback   = zeros(N, T);            % one-step predictive feedback mean
prevCtxProb    = zeros(T, Cwidth);       % local/modal predicted context probs (before y)
postCtxProb    = zeros(T, Cwidth);       % local/modal context responsibilities (after y)

fprintf('Running 500-trial multi-dimensional COIN example (N=%d)...\n', N);
tic;
for t = 1:T
    coin.observe_q(cues(t));

    % "Prev": predictive distributions before seeing the observation.
    predFeedback(:, t) = coin.predictive_feedback_moments(cues(t));
    pv = coin.predicted_context_probabilities_local();
    prevCtxProb(t, 1:numel(pv)) = pv;

    % Process the observation.
    coin.observe_y(feedbacks(:, t));

    % "Post": posterior state and context responsibilities after the update.
    postStateMean(:, t) = coin.state_moments();
    rv = coin.context_responsibilities_local();
    postCtxProb(t, 1:numel(rv)) = rv;
end
elapsed = toc;
fprintf('Processed %d trials in %.2f s (%.1f ms/trial).\n', T, elapsed, 1e3*elapsed/T);

% Effective number of instantiated contexts per trial: contexts carrying appreciable
% posterior responsibility at least once in and before current trial
nContexts = sum(cummax(postCtxProb > 0.05, 1), 2).';


% -------------------------------------------------------------------------
% End-of-run confirmation: learned per-context parameters.
% -------------------------------------------------------------------------
D = coin.diagnostics();
K = D.K;
fprintf('\n================ Learned context structure ================\n');
fprintf('Discovered %d context(s) (modal cardinality).\n', K);
[~, dominantCue] = max(D.cue_prob, [], 2);     % most associated cue per context
for c = 1:K
    A = D.A(:, :, c);
    d = D.drift(:, c);
    b = D.bias(:, c);
    sm = D.state_mean(:, c);
    sd = sqrt(max(diag(D.state_cov(:, :, c)), 0));
    fprintf('\n--- Context %d ---\n', c);
    fprintf('  retention A      = [% .3f % .3f ; % .3f % .3f]\n', A(1,1), A(1,2), A(2,1), A(2,2));
    fprintf('  spectral radius  = %.3f (stable if < 1)\n', max(abs(eig(A))));
    fprintf('  drift d          = [% .3f % .3f]\n', d(1), d(2));
    fprintf('  bias b           = [% .3f % .3f]\n', b(1), b(2));
    fprintf('  state mean +/-sd = [% .3f % .3f] +/- [% .3f % .3f]\n', sm(1), sm(2), sd(1), sd(2));
    fprintf('  most-likely cue  = %d  (p = %.2f)\n', dominantCue(c), D.cue_prob(c, dominantCue(c)));
    fprintf('  predicted prob   = %.3f , responsibility = %.3f\n', ...
        D.predicted_probabilities(c), D.responsibilities(c));
end

fprintf('\nFinal predicted context probabilities (prev): ');
fprintf('%.3f ', D.predicted_probabilities); fprintf('\n');
fprintf('Final context responsibilities      (post): ');
fprintf('%.3f ', D.responsibilities); fprintf('\n');

% sampled_context_count returns the distribution over which context index the
% modal particles currently occupy (one entry per context slot).
countDist = coin.sampled_context_count();
fprintf('Sampled-context occupancy distribution     : ');
fprintf('%.3f ', countDist); fprintf('\n');

fprintf('\nTrue block targets (for reference):\n');
for b = 1:numel(blockLens)
    fprintf('  block %d (cue %d): target = [% .3f % .3f]\n', b, blockCues(b), targets(1,b), targets(2,b));
end
fprintf('===========================================================\n');

% -------------------------------------------------------------------------
% Plots
% -------------------------------------------------------------------------
blockEdges = bnd(2:end-1);   % trial indices where the contingency switches

% Figure 1: per-dimension state tracking.
figure('Name', 'MD COIN: state tracking');
for dim = 1:N
    subplot(N, 1, dim);
    plot(1:T, feedbacks(dim, :), '.', 'Color', [0.7 0.7 0.7]); hold on;
    plot(1:T, trueState(dim, :), 'k--', 'LineWidth', 1.2);
    plot(1:T, postStateMean(dim, :), 'b-', 'LineWidth', 1.2);
    plot(1:T, predFeedback(dim, :), 'r:', 'LineWidth', 1.0);
    yl = ylim;
    for e = blockEdges
        plot([e e], yl, 'Color', [0.5 0.5 0.5 0.4]);
    end
    xlabel('Trial'); ylabel(sprintf('State dim %d', dim));
    legend('Observed', 'True target', 'Posterior mean', 'Predicted feedback', 'Location', 'best');
end

% Figure 2: context probabilities over trials (prev and post).
figure('Name', 'MD COIN: context probabilities');
subplot(2, 1, 1);
area(1:T, prevCtxProb(:, 1:max(K, 1)));
xlabel('Trial'); ylabel('Predicted prob (prev)');
title('Local/modal predicted context probabilities (before observation)');
ylim([0 1]);
subplot(2, 1, 2);
area(1:T, postCtxProb(:, 1:max(K, 1)));
xlabel('Trial'); ylabel('Responsibility (post)');
title('Local/modal context responsibilities (after observation)');
ylim([0 1]);
legend(arrayfun(@(k) sprintf('Context %d', k), 1:max(K,1), 'UniformOutput', false), ...
    'Location', 'eastoutside');

% Figure 3: 2-D state-space trajectory with the true targets.
figure('Name', 'MD COIN: 2-D trajectory');
plot(postStateMean(1, :), postStateMean(2, :), '-', 'Color', [0.2 0.2 0.8]); hold on;
plot(trueState(1, :), trueState(2, :), 'ks', 'MarkerFaceColor', 'k');
plot(targets(1, :), targets(2, :), 'rp', 'MarkerSize', 14, 'MarkerFaceColor', 'r');
xlabel('State dim 1'); ylabel('State dim 2');
title('Posterior state trajectory and true targets');
legend('Posterior mean path', 'True (per trial)', 'Block targets', 'Location', 'best');
axis equal;

% Figure 4: effective number of contexts over trials.
figure('Name', 'MD COIN: context count');
plot(1:T, nContexts, 'LineWidth', 1.2);
xlabel('Trial'); ylabel('Effective # contexts');
title('Number of sampled contexts over the experiment');
ylim([0 maxContexts + 1]);

% =========================================================================
% Grid-based predictive densities (multi-dimensional query API)
% =========================================================================
% The four methods below all evaluate Gaussian-mixture posteriors/predictives
% (mixed over particles and contexts) at a set of query points. In the N-D
% model a "grid" is an N-by-K matrix whose COLUMNS are the points to evaluate;
% each method returns a 1-by-K density row (or, for the CDF, an N-by-1 vector
% of marginal probabilities). Here N = 2, so we evaluate over a 2-D plane and
% draw the resulting densities as filled contour maps.
%
% These are a SNAPSHOT taken at the end of the 500-trial run, i.e. they
% summarise the model's beliefs given everything it has seen. The predictive
% (feedback / CDF) quantities are previewed for one hypothetical next trial
% under the recall cue (cue 1); they are read-only and do not advance the model.
demoCue = 1;                              % preview the next trial under cue 1
coin.observe_q(demoCue);                  % register the cue (read-only previews follow)

% Moments used only to centre/scale the evaluation grids sensibly.
[muS, covS]   = coin.state_moments();                       % posterior latent state
[muF, SigmaF] = coin.predictive_feedback_moments(demoCue);  % predictive feedback
sdS = sqrt(max(diag(covS), 0));
sdF = sqrt(max(diag(SigmaF), 0));

% Helper: build a 2-D tensor grid centred at "c" spanning +/- "span" std and
% return both the plotting meshes (gx, gy axes; GX, GY) and the N-by-K column
% list of query points expected by the density methods.
ng = 90;                                  % grid resolution per axis
makeAxes = @(c, sd, span) linspace(c - span*sd, c + span*sd, ng);

% --- Figure 5: posterior latent-state density, p(state | all data) --------
% state_probability marginalises over particles and contexts using the
% posterior responsibilities and filtered (posterior) per-context moments.
axSx = makeAxes(muS(1), max(sdS(1), 1e-3), 4);
axSy = makeAxes(muS(2), max(sdS(2), 1e-3), 4);
[GXs, GYs] = meshgrid(axSx, axSy);
ptsS = [GXs(:)'; GYs(:)'];                          % 2-by-K query points
Zstate = reshape(coin.state_probability(ptsS), size(GXs));

figure('Name', 'MD COIN: posterior state density');
contourf(axSx, axSy, Zstate, 20, 'LineColor', 'none'); hold on;
colorbar; colormap('parula');
plot(muS(1), muS(2), 'w+', 'MarkerSize', 12, 'LineWidth', 2);       % posterior mean
plot(targets(1, end), targets(2, end), 'rp', ...                    % current true target
    'MarkerSize', 16, 'MarkerFaceColor', 'r');
xlabel('State dim 1'); ylabel('State dim 2');
title('Posterior latent-state density  p(state \mid data)');
legend('density', 'posterior mean', 'true target', 'Location', 'best');
axis tight;

% --- Figure 6: predictive feedback density for the next (cue-1) trial ------
% state_feedback_probability uses the PREDICTED context probabilities and the
% predictive feedback moments (state inflated by the observation noise R),
% i.e. the model's belief about the observation it is about to receive.
axFx = makeAxes(muF(1), max(sdF(1), 1e-3), 4);
axFy = makeAxes(muF(2), max(sdF(2), 1e-3), 4);
[GXf, GYf] = meshgrid(axFx, axFy);
ptsF = [GXf(:)'; GYf(:)'];
Zfb = reshape(coin.state_feedback_probability(ptsF), size(GXf));

figure('Name', 'MD COIN: predictive feedback density');
contourf(axFx, axFy, Zfb, 20, 'LineColor', 'none'); hold on;
colorbar;
plot(muF(1), muF(2), 'w+', 'MarkerSize', 12, 'LineWidth', 2);       % predictive mean
plot(targets(1, end), targets(2, end), 'rp', ...
    'MarkerSize', 16, 'MarkerFaceColor', 'r');
xlabel('Feedback dim 1'); ylabel('Feedback dim 2');
title('Predictive feedback density  p(y_{next} \mid data, cue 1)');
legend('density', 'predictive mean', 'true target', 'Location', 'best');
axis tight;

% --- Figure 7: per-context posterior state densities ----------------------
% state_given_context_probability returns a containers.Map keyed by GLOBAL
% context label; each value is that single context's posterior state density
% over the query points. We overlay one contour set per context so the
% spatial separation of the learned contexts is visible. We reuse the latent
% state grid from Figure 5 (it is centred on the full posterior).
ctxDens = coin.state_given_context_probability(ptsS);
ctxKeys = cell2mat(ctxDens.keys);
cols = lines(max(numel(ctxKeys), 1));

figure('Name', 'MD COIN: per-context state densities'); hold on;
legEntries = cell(1, numel(ctxKeys));
for i = 1:numel(ctxKeys)
    c = ctxKeys(i);
    Zc = reshape(ctxDens(c), size(GXs));
    % Draw two contour levels at fixed fractions of each context's own peak,
    % so contexts of differing sharpness remain comparable on the plot.
    peak = max(Zc(:));
    if peak <= 0
        continue;                          % empty/degenerate context, skip
    end
    contour(axSx, axSy, Zc, peak * [0.25 0.6], 'LineColor', cols(i, :), 'LineWidth', 1.5);
    legEntries{i} = sprintf('Context %d', c);
end
% Mark the true block targets the model was trained on, for reference.
plot(targets(1, :), targets(2, :), 'kp', 'MarkerSize', 14, 'MarkerFaceColor', 'k');
xlabel('State dim 1'); ylabel('State dim 2');
title('Per-context posterior state densities  p(state \mid context)');
legEntries{end+1} = 'block targets';
legend(legEntries(~cellfun(@isempty, legEntries)), 'Location', 'best');
axis tight;

% --- Figure 8: marginal predictive CDF of the next feedback ---------------
% predictive_state_feedback_cdf(y, q) returns, in N-D, the per-dimension
% marginal CDF vector p_j = P(Y_j <= y_j) for the next feedback under cue q.
% We sweep each dimension's threshold across its predictive range and read off
% that dimension's marginal CDF, giving the familiar S-shaped distribution
% function for each output dimension. (Each marginal is itself a 1-D Gaussian
% mixture over particles/contexts, hence the slight departures from a single
% Gaussian ogive.)
nSweep = 200;
figure('Name', 'MD COIN: marginal predictive CDF');
for dim = 1:N
    sweep = linspace(muF(dim) - 4*sdF(dim), muF(dim) + 4*sdF(dim), nSweep);
    cdfVals = zeros(1, nSweep);
    for k = 1:nSweep
        % Hold the other dimension at its predictive mean; only entry "dim" of
        % the returned vector is used, and it depends solely on y(dim).
        yq = muF;
        yq(dim) = sweep(k);
        pvec = coin.predictive_state_feedback_cdf(yq, demoCue);
        cdfVals(k) = pvec(dim);
    end
    subplot(N, 1, dim);
    plot(sweep, cdfVals, 'b-', 'LineWidth', 1.4); hold on;
    plot([muF(dim) muF(dim)], [0 1], 'k--');       % predictive mean (CDF ~ 0.5)
    yl = ylim;
    plot([trueState(dim, end) trueState(dim, end)], yl, 'r:', 'LineWidth', 1.2);  % true target
    xlabel(sprintf('Feedback threshold y_%d', dim));
    ylabel(sprintf('P(Y_%d \\leq y_%d)', dim, dim));
    title(sprintf('Marginal predictive CDF, dimension %d (cue %d)', dim, demoCue));
    ylim([0 1]);
    legend('marginal CDF', 'predictive mean', 'true target', 'Location', 'southeast');
end
