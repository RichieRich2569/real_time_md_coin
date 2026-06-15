function updateLocalCueMatrix(obj)
    Cmax = obj.max_contexts + 1;
    P = obj.num_particles;
    Qn = max(1, size(obj.D.global_cue_probabilities, 1));
    L = zeros(Cmax, Qn, P);
    for p = 1:P
        counts = obj.D.n_cue(:,1:min(Qn,size(obj.D.n_cue,2)),p);
        if size(counts,2) < Qn
            counts(:,end+1:Qn) = 0;
        end
        raw = counts + obj.alpha_cue .* obj.D.global_cue_probabilities(1:Qn,p)';
        valid = false(Cmax,1);
        valid(1:obj.D.C(p)) = true;
        if obj.D.C(p) < obj.max_contexts
            valid(obj.D.C(p)+1) = true;
        end
        raw(~valid,:) = 0;
        rowSums = sum(raw, 2);
        for c = 1:Cmax
            if rowSums(c) > 0
                L(c,:,p) = raw(c,:) ./ rowSums(c);
            end
        end
    end
    obj.D.local_cue_matrix = L;
end
