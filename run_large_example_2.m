% Run example over 500 trials, similarly to coin-rl tests
clear; clc;

% Add path to this directory if necessary
if ~exist('RealTimeCOIN', 'class')
    addpath(fileparts(mfilename('fullpath')));
end

% Initialise COIN
mean_retention = 0.9425;
precision_drift = 1/(1-mean_retention)^2;
coin = RealTimeCOIN(sigma_sensory_noise=0.003, sigma_motor_noise=0.00182, prior_mean_retention=mean_retention, prior_precision_drift=precision_drift);

true_perturbations = [2*ones(1,300), zeros(1,100) ones(1,50) -ones(1,50) -2*ones(1,50)];
observations = true_perturbations + coin.sigma_sensory_noise * randn(size(true_perturbations));
cues = [ones(1,300) 2*ones(1,100) ones(1,50) -ones(1,50) -2*ones(1,50)];

% Prepare values for plotting
predicted_means = zeros(size(observations));
ctxWidth = coin.max_contexts + 1;
ctx_probs = zeros(length(observations), ctxWidth);
pred_ctx_probs = zeros(length(observations), ctxWidth);
state_probabilities = zeros(length(observations), 101); % for grid of 101 points

grid = linspace(-3.0, 3.0, 101);

tic;
for t = 1:length(observations)
    coin.observe_y(observations(t));
    probs = coin.context_responsibilities_local();
    ctx_probs(t, 1:numel(probs)) = probs;

    pred_probs = coin.predicted_context_probabilities_local();
    pred_ctx_probs(t, 1:numel(pred_probs)) = pred_probs;
    % compute predicted state mean and variance
    dens = coin.state_probability(grid);
    state_probabilities(t, :) = dens;
    % normalise density
    area = trapz(grid, dens);
    if area > 0
        dens = dens / area;
    end
    mean_state = trapz(grid, dens .* grid);
    predicted_means(t) = mean_state;
end
elapsed_time = toc;
fprintf('Time taken for %d trials: %.2f seconds\n', length(observations), elapsed_time);

% Plot predicted state means over trials as well as observations and true value
figure;
plot(1:length(observations), predicted_means, '-o');
hold on;
plot(1:length(observations), observations, 'x');
plot(1:length(observations), true_perturbations, '--');
xlabel('Trial');
ylabel('State Value');
legend('Predicted', 'Observed', 'True');

% Plot context probabilities over trials
figure;
plot(1:length(observations), ctx_probs);
xlabel('Trial');
ylabel('Local/modal Context Probability');
legend(arrayfun(@(k) sprintf('Local %d', k), 1:ctxWidth, 'UniformOutput', false));

% Plot predicted context probabilities over trials
figure;
plot(1:length(observations), pred_ctx_probs);
xlabel('Trial');
ylabel('Local/modal Predicted Context Probability');
legend(arrayfun(@(k) sprintf('Local %d', k), 1:ctxWidth, 'UniformOutput', false));

% Plot final exact state probabilities given globally aligned context.
state_given_ctx_container = coin.state_given_context_probability(grid);
keys = state_given_ctx_container.keys;
figure;
hold on;
for k = 1:length(keys)
    plot(grid, state_given_ctx_container(keys{k}), 'LineWidth', 1.2);
end
xlabel('State Value');
ylabel('Density');
title('Final state probability given globally aligned context');
legend(arrayfun(@(k) sprintf('Context %d', k), cell2mat(keys), 'UniformOutput', false));

