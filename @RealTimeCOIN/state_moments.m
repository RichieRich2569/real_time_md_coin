function [mu, v] = state_moments(obj)
    W = obj.D.predicted_probabilities;
    mu = sum(W .* obj.D.state_mean, 'all') ./ obj.num_particles;
    second = sum(W .* (obj.D.state_var + obj.D.state_mean.^2), 'all') ./ obj.num_particles;
    v = max(second - mu.^2, 0);
end
