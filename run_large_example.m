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
ctx_probs = zeros(length(observations), coin.max_contexts);
state_probabilities = zeros(length(observations), 101); % for grid of 101 points
state_given_ctx_probs = zeros(length(observations), coin.max_contexts, 101); % for grid of 101 points

grid = linspace(-1.5, 1.5, 101);

tic;
for t = 1:length(observations)
    coin.observe_y(observations(t));
    probs = coin.context_probabilities();
    ctx_probs(t, 1:length(probs)) = cell2mat(probs.values);
    % compute predicted state mean and variance
    dens = coin.state_probability(grid);
    state_probabilities(t, :) = dens;
    state_given_ctx_container = coin.state_given_context_probability(grid); % Returns containers.Map
    % Print keys in container
    keys = state_given_ctx_container.keys;
    for k = 1:length(keys)
        state_given_ctx_probs(t, keys{k}, :) = state_given_ctx_container(keys{k});
    end
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
ylabel('Context Probability');
legend(arrayfun(@(k) sprintf('Context %d', k), 1:coin.max_contexts, 'UniformOutput', false));

% Plot state probabilities given context over trials for each context as heatmap with different colours for different
% contexts in a single axis
figure;
colors = lines(coin.max_contexts);
combined_rgb = zeros(length(grid), length(observations), 3);
for k = 1:coin.max_contexts
    prob_data = squeeze(state_given_ctx_probs(:, k, :))';
    % Normalize to [0, 1] for this context
    prob_max = max(prob_data(:));
    if prob_max > 0
        prob_data = prob_data / prob_max;
    end
    % Add weighted contribution to combined image
    for c = 1:3
        combined_rgb(:, :, c) = combined_rgb(:, :, c) + prob_data * colors(k, c);
    end
end
% Normalize combined image to [0, 1]
combined_rgb = combined_rgb / max(combined_rgb(:));
imagesc(1:length(observations), grid, combined_rgb);
set(gca, 'YDir', 'normal');
xlabel('Trial');
ylabel('State Value');
title('State Probability Given Context');
colorbar;

