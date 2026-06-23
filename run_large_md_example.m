%RUN_LARGE_MD_EXAMPLE Long-run demonstration of the multi-dimensional RealTimeCOIN.
%
%   Runs the N-dimensional model for 500 trials over a sequence of distinct
%   2-D "constant contingencies" (each a different context), added with cues and each with its own latent
%   target, including a recall block, to demonstrate multi-context adaptation.
%   It exercises and reports every major element of the implementation:
%
%     * State tracking      - posterior state mean vs. true target vs. noisy
%                             observation, per dimension and in 2-D.
%     * Context prediction  - predicted_context_probabilities() ("prev",
%                             before each observation).
%     * Context responsibility - context_responsibilities() ("post", after
%                             each observation).
%     * Learned parameters  - per-context retention A, drift d, bias b, the
%                             filtered state mean/covariance and the cue
%                             association, read from diagnostics() at the end.
%
%   Companion to run_large_example.m (scalar). Multi-dimensional summaries use
%   the MD-capable query API; grid-based densities remain scalar-only.

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
feedbacks = trueState + obsSigma * randn(N, T);

% -------------------------------------------------------------------------
% Run the model, recording everything needed for confirmation.
% -------------------------------------------------------------------------
postStateMean  = zeros(N, T);            % posterior E[state] each trial
predFeedback   = zeros(N, T);            % one-step predictive feedback mean
prevCtxProb    = zeros(T, Cwidth);       % predicted context probs (before y)
postCtxProb    = zeros(T, Cwidth);       % context responsibilities (after y)

fprintf('Running 500-trial multi-dimensional COIN example (N=%d)...\n', N);
tic;
for t = 1:T
    coin.observe_q(cues(t));

    % "Prev": predictive distributions before seeing the observation.
    predFeedback(:, t) = coin.predictive_feedback_moments(cues(t));
    pv = coin.predicted_context_probabilities();
    prevCtxProb(t, 1:numel(pv)) = pv;

    % Process the observation.
    coin.observe_y(feedbacks(:, t));

    % "Post": posterior state and context responsibilities after the update.
    postStateMean(:, t) = coin.state_moments();
    respMap = coin.context_responsibilities();
    ks = respMap.keys;
    for i = 1:numel(ks)
        postCtxProb(t, ks{i}) = respMap(ks{i});
    end
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
title('Predicted context probabilities (before observation)');
ylim([0 1]);
subplot(2, 1, 2);
area(1:T, postCtxProb(:, 1:max(K, 1)));
xlabel('Trial'); ylabel('Responsibility (post)');
title('Context responsibilities (after observation)');
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
