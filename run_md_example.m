%RUN_MD_EXAMPLE Demonstration of the multi-dimensional RealTimeCOIN extension.
%
%   Companion to run_example.m for the N-dimensional state extension
%   (state_dim > 1). It builds a 2-D RealTimeCOIN with correlated process and
%   observation noise, streams a cue/feedback sequence that switches between
%   two latent targets, and after each trial reports the one-step predictive
%   feedback distribution and the posterior state moments.
%
%   Multi-dimensional summaries are obtained from state_moments() (which
%   returns an N-by-1 mean vector and an N-by-N covariance for state_dim > 1)
%   and predictive_feedback_moments(). The context-alignment and grid-based
%   query methods (context_responsibilities, state_probability, diagnostics)
%   remain scalar-only and are deferred to a later phase.

clear; clc;

% Add path to this directory if necessary
if ~exist('RealTimeCOIN', 'class')
    addpath(fileparts(mfilename('fullpath')));
end

rng(7);

N = 2;  % state (and, with the identity observation map, observation) dimension

% Correlated process/observation noise covariances supplied as full matrices.
% Leaving these empty would fall back to the isotropic defaults derived from
% the scalar sigma_* properties (sigma_process_noise^2 * I, etc.).
Q = [2.5e-3, 1.0e-3; 1.0e-3, 2.0e-3];
R = [9.0e-3, -2.0e-3; -2.0e-3, 1.2e-2];

coin = RealTimeCOIN('num_particles', 300, 'max_contexts', 4, 'state_dim', N, ...
    'infer_bias', true, 'process_noise_covariance', Q, ...
    'observation_noise_covariance', R);

% Cue sequence and the 2-D latent target associated with each cue. The target
% switches partway through the block to exercise multi-context inference.
cues = [1 1 1 2 2 2 2 1 1 2];
targets = containers.Map('KeyType', 'double', 'ValueType', 'any');
targets(1) = [ 0.5; -0.2];
targets(2) = [-0.3;  0.6];

T = numel(cues);
true_states = zeros(N, T);
feedbacks = zeros(N, T);
Lr = chol(R, 'lower');
for t = 1:T
    true_states(:, t) = targets(cues(t));
    feedbacks(:, t) = true_states(:, t) + Lr * randn(N, 1);   % add observation noise
end

% Storage for plotting.
pred_feedback_mean = zeros(N, T);
pred_feedback_std = zeros(N, T);   % per-dimension predictive 1-sigma
post_state_mean = zeros(N, T);

fprintf('Starting multi-dimensional real-time COIN example (N=%d)...\n', N);
for t = 1:T
    coin.observe_q(cues(t));

    % One-step predictive feedback distribution BEFORE seeing the observation.
    [mu_pred, Sigma_pred] = coin.predictive_feedback_moments(cues(t));
    pred_feedback_mean(:, t) = mu_pred;
    pred_feedback_std(:, t) = sqrt(max(diag(Sigma_pred), 0));

    % Process the observation.
    coin.observe_y(feedbacks(:, t));

    % Posterior predictive state moments AFTER the update.
    [mu_state, cov_state] = coin.state_moments();
    post_state_mean(:, t) = mu_state;

    fprintf('\nTrial %2d (cue %d):\n', t, cues(t));
    fprintf('  Predicted feedback mean : [% .3f % .3f]\n', mu_pred(1), mu_pred(2));
    fprintf('  Observed feedback       : [% .3f % .3f]\n', feedbacks(1, t), feedbacks(2, t));
    fprintf('  Posterior state mean    : [% .3f % .3f]\n', mu_state(1), mu_state(2));
    fprintf('  Posterior state std     : [% .3f % .3f]\n', sqrt(cov_state(1, 1)), sqrt(cov_state(2, 2)));
end

fprintf('\nFinal trial count: %d\n', coin.Trial);

% --- Plot 1: per-dimension predicted vs observed vs true over trials ---
figure('Name', 'Multi-dim RealTimeCOIN: state dimensions');
for dim = 1:N
    subplot(N, 1, dim);
    % Predictive +/-1 sigma band around the one-step prediction.
    fill([1:T, T:-1:1], ...
        [pred_feedback_mean(dim, :) + pred_feedback_std(dim, :), ...
        fliplr(pred_feedback_mean(dim, :) - pred_feedback_std(dim, :))], ...
        [0.8 0.8 1], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    hold on;
    plot(1:T, post_state_mean(dim, :), '-o');
    plot(1:T, feedbacks(dim, :), 'x');
    plot(1:T, true_states(dim, :), '--');
    xlabel('Trial');
    ylabel(sprintf('State dim %d', dim));
    legend('Predictive \pm1\sigma', 'Posterior mean', 'Observed', 'True', 'Location', 'best');
end

% --- Plot 2: 2-D state-space trajectory (when N == 2) ---
if N == 2
    figure('Name', 'Multi-dim RealTimeCOIN: 2-D trajectory');
    plot(post_state_mean(1, :), post_state_mean(2, :), '-o');
    hold on;
    plot(feedbacks(1, :), feedbacks(2, :), 'x');
    plot(true_states(1, :), true_states(2, :), 's');
    xlabel('State dim 1');
    ylabel('State dim 2');
    legend('Posterior mean', 'Observed', 'True', 'Location', 'best');
    title('2-D state-space trajectory');
    axis equal;
end
