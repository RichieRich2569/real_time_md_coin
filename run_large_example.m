% Run example over 500 trials, similarly to original coin tests
clear; clc;

% Add path to this directory if necessary
if ~exist('RealTimeCOIN', 'class')
    addpath(fileparts(mfilename('fullpath')));
end

% Initialise COIN
coin = RealTimeCOIN();

true_perturbations = [zeros(1,50), ones(1,125) -ones(1,15) NaN(1,150)];
observations = true_perturbations + coin.sigma_sensory_noise * randn(size(true_perturbations));

% Prepare values for plotting
predicted_means = zeros(size(observations));
ctxWidth = coin.max_contexts + 1;
ctx_probs = zeros(length(observations), ctxWidth);
state_probabilities = zeros(length(observations), 101); % for grid of 101 points

grid = linspace(-1.5, 1.5, 101);

tic;
for t = 1:length(observations)
    coin.observe_y(observations(t));
    probs = coin.context_responsibilities_local();
    ctx_probs(t, 1:numel(probs)) = probs;
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

