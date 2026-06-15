function updateLocalTransitionMatrix(obj)
    Cmax = obj.max_contexts + 1;
    P = obj.num_particles;
    kappa = obj.kappa();
    T = zeros(Cmax, Cmax, P);
    for p = 1:P
        raw = obj.nContextSlice(p) + obj.alpha_context .* obj.D.global_transition_probabilities(:,p)' + kappa .* eye(Cmax);
        valid = false(1, Cmax);
        valid(1:obj.D.C(p)) = true;
        if obj.D.C(p) < obj.max_contexts
            valid(obj.D.C(p)+1) = true;
        end
        raw(:, ~valid) = 0;
        raw(~valid, :) = 0;
        rowSums = sum(raw, 2);
        for r = 1:Cmax
            if rowSums(r) > 0
                T(r,:,p) = raw(r,:) ./ rowSums(r);
            end
        end
    end
    obj.D.local_transition_matrix = T;
end
