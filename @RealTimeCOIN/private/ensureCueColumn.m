function ensureCueColumn(obj, q)
    if isempty(q)
        return;
    end
    Cmax = obj.max_contexts + 1;
    P = obj.num_particles;
    if size(obj.D.n_cue, 2) < q
        extra = q - size(obj.D.n_cue, 2);
        obj.D.n_cue(:, end+1:end+extra, :) = zeros(Cmax, extra, P);
    end
    if size(obj.D.global_cue_probabilities, 1) < q
        extra = q - size(obj.D.global_cue_probabilities, 1);
        obj.D.global_cue_probabilities(end+1:end+extra, :) = 0;
    end
    if size(obj.D.local_cue_matrix, 2) < q
        extra = q - size(obj.D.local_cue_matrix, 2);
        obj.D.local_cue_matrix(:, end+1:end+extra, :) = zeros(Cmax, extra, P);
    end
end
