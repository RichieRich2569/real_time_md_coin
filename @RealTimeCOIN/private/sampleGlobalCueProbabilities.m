function sampleGlobalCueProbabilities(obj)
    if size(obj.D.global_cue_probabilities, 1) == 0
        return;
    end
    P = obj.num_particles;
    Qn = max(obj.D.Q + 1, size(obj.D.global_cue_probabilities, 1));
    obj.ensureCueColumn(Qn);
    for p = 1:P
        counts = obj.D.n_cue(:,1:Qn,p);
        base = repmat(obj.alpha_cue .* obj.D.global_cue_probabilities(1:Qn,p)', size(counts,1), 1);
        m = obj.sample_num_tables(base, counts);
        alpha = sum(m, 1);
        alpha(obj.D.Q + 1) = obj.gamma_cue;
        if obj.D.Q + 2 <= Qn
            alpha(obj.D.Q + 2:end) = 0;
        end
        obj.D.global_cue_probabilities(1:Qn,p) = obj.dirichletSample(alpha(:));
    end
end
