%RUN_LARGE_MD_EXAMPLE Long-run demonstration of the multi-dimensional RealTimeCOIN.
%
%   Runs the N-dimensional model for 500 trials over a sequence of distinct
%   4-D "constant contingencies" (each a different context), no cues and each with its own latent
%   target variation.

clear; clc;

% Add path to this directory if necessary
if ~exist('RealTimeCOIN', 'class')
    addpath(fileparts(mfilename('fullpath')));
end

rng(11);

N = 4;                                   % state / observation dimensionality
maxContexts = 10;
Cwidth = maxContexts + 1;                % context-vector width (incl. novel slot)

coin = RealTimeCOIN('num_particles', 200, 'max_contexts', maxContexts, ...
    'state_dim', N, 'infer_bias', true);

% -------------------------------------------------------------------------
% Build the experiment: a sequence of 4-D contingencies. Each context has a slight variation:
% 1. Context 1 is a fixed value (0.5, -0.5, 0.5, -0.5).
% 2. Context 2 is a slightly drifting value through trials, starting at (-0.5, 0.5, 0.5, -0.5) and drifting by (0.01, -0.01, 0.01, -0.01) per trial.
% 3. Context 3 implements swapping values (retention matrix with off-diagonal elements) to create a more complex dynamic. Every trial, it swaps the first two dimensions with a starting value of (0.4, 0.2, -0.4, -0.4).
% -------------------------------------------------------------------------
blockLens   = [150 150 130 70];          % trial counts (sum = 500)

T = sum(blockLens);
trueState = zeros(N, T);
bnd = [0 cumsum(blockLens)];

% Context 1: fixed target
trueState(:, 1:bnd(2)) = repmat([0.5; -0.5; 0.5; -0.5], 1, blockLens(1));

% Context 2: drifting target
drift = [0.01; -0.01; 0.01; -0.01];
for t = (bnd(2)+1):bnd(3)
    trueState(:, t) = [-0.5; 0.5; 0.5; -0.5] + drift * (t - bnd(2));
end

% Context 3: swapping target
swapA = [0 1 0 0; 1 0 0 0; 0 0 1 0; 0 0 0 1]; % swaps first two dimensions
for t = (bnd(3)+1):bnd(4)
    if t == bnd(3)+1
        trueState(:, t) = [0.4; 0.2; -0.4; -0.4];
    else
        trueState(:, t) = swapA * trueState(:, t-1);
    end
end

% Back to first context
trueState(:, (bnd(4)+1):bnd(5)) = repmat([0.5; -0.5; 0.5; -0.5], 1, blockLens(4));

% Noisy observations, zero covariance but different noise values for each dimension.
Q = diag([0.05, 0.1, 0.15, 0.2]); % different noise levels for each dimension
feedbacks = trueState + chol(Q)' * randn(N, T);

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
    % "Prev": predictive distributions before seeing the observation.
    predFeedback(:, t) = coin.predictive_feedback_moments();
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

    fprintf("Finished trial %d/%d\n", t, T);
    fprintf("Elapsed time: %.2f s (%.1f ms/trial)\n", toc, 1e3*toc/t);
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

% Figure 3: effective number of contexts over trials.
figure('Name', 'MD COIN: context count');
plot(1:T, nContexts, 'LineWidth', 1.2);
xlabel('Trial'); ylabel('Effective # contexts');
title('Number of sampled contexts over the experiment');
ylim([0 maxContexts + 1]);
