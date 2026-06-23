function densities = state_feedback_probability(obj, values)
    obj.assertScalarOnly('state_feedback_probability');
    values = values(:)';
    densities = zeros(size(values));
    W = obj.D.predicted_probabilities;
    M = obj.D.state_feedback_mean;
    V = obj.D.state_feedback_var;
    for p = 1:obj.num_particles
        for c = 1:(obj.max_contexts+1)
            if W(c,p) > 0
                densities = densities + W(c,p) .* obj.normal_pdf(values, M(c,p), V(c,p));
            end
        end
    end
    densities = densities ./ obj.num_particles;
end
