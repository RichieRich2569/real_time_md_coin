%RUN_EXAMPLE Demonstration of the RealTimeCOIN class.
%
%   This script creates a RealTimeCOIN object with a modest number of
%   particles, processes a sequence of cues and observations and
%   prints context probabilities and state estimates after each trial.

clear; clc;

% Add path to this directory if necessary
if ~exist('RealTimeCOIN', 'class')
    addpath(fileparts(mfilename('fullpath')));
end

% Initialise with 200 particles, allow up to 5 contexts, infer bias
coin = RealTimeCOIN('num_particles', 200, 'max_contexts', 5, 'infer_bias', true);

% Example cue and feedback sequences
cues = [1,1,2,2,1,3,1];
obs_noise = coin.sigma_sensory_noise;

true_values = [0.2, 0.2, 0.5, 0.5, 0.2, -0.1, 0.2]; % for reference only
feedbacks = true_values + obs_noise * randn(size(true_values)); % add some noise

grid = linspace(-1.5, 1.5, 101);

% Prepare values for plotting
predicted_means = zeros(size(cues));

fprintf('Starting real‑time COIN example...\n');
for t = 1:length(cues)
    fprintf('\nTrial %d:\n', t);
    coin.observe_q(cues(t));
    coin.observe_y(feedbacks(t));
    probs = coin.context_probabilities();
    keys = probs.keys;
    for k = 1:length(keys)
        fprintf('  Context %d probability: %.3f\n', keys{k}, probs(keys{k}));
    end
    % compute predicted state mean and variance
    dens = coin.state_probability(grid);
    % normalise density
    area = trapz(grid, dens);
    if area > 0
        dens = dens / area;
    end
    mean_state = trapz(grid, dens .* grid);
    predicted_means(t) = mean_state;
    var_state = trapz(grid, dens .* (grid - mean_state).^2);
    fprintf('  Predicted state mean: %.3f, variance: %.4f\n', mean_state, var_state);
end

fprintf('\nFinal trial count: %d\n', coin.Trial);

% Plot predicted state means over trials as well as observations and true value
figure;
plot(1:length(cues), predicted_means, '-o');
hold on;
plot(1:length(cues), feedbacks, 'x');
plot(1:length(cues), true_values, '--');
xlabel('Trial');
ylabel('State Value');
legend('Predicted', 'Observed', 'True');