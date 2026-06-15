function instantiateCueIfNeeded(obj, q)
    if isempty(q) || q <= obj.D.Q
        return;
    end
    obj.ensureCueColumn(q + 1);
    b = obj.betaSample(ones(1, obj.num_particles), obj.gamma_cue * ones(1, obj.num_particles));
    mass = obj.D.global_cue_probabilities(q, :);
    obj.D.global_cue_probabilities(q+1, :) = mass .* (1 - b);
    obj.D.global_cue_probabilities(q, :) = mass .* b;
    obj.D.Q = q;
end
