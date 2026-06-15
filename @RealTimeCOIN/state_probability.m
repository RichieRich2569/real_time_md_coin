function densities = state_probability(obj, values)
    values = values(:)';
    densities = zeros(size(values));
    W = obj.D.responsibilities;
    M = obj.D.state_filtered_mean;
    V = obj.D.state_filtered_var;
    for p = 1:obj.num_particles
        for c = 1:(obj.max_contexts+1)
            if W(c,p) > 0
                densities = densities + W(c,p) .* obj.normal_pdf(values, M(c,p), V(c,p));
            end
        end
    end
    densities = densities ./ obj.num_particles;
end
