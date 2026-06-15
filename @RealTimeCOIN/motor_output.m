function u = motor_output(obj)
    W = obj.D.predicted_probabilities;
    u = sum(W .* obj.D.state_feedback_mean, 'all') ./ obj.num_particles;
end
