function predictContext(obj, q)
    obj.updateLocalTransitionMatrix();
    P = obj.num_particles;
    Cmax = obj.max_contexts + 1;
    D = obj.D; %#ok<*PROPLC>
    prior = zeros(Cmax, P);
    for p = 1:P
        prior(:,p) = D.local_transition_matrix(D.context(p), :, p)';
    end
    obj.D.prior_probabilities = obj.normalizeColumns(prior);

    if isempty(q)
        obj.D.probability_cue = ones(Cmax, P);
        obj.D.predicted_probabilities = obj.D.prior_probabilities;
    else
        obj.updateLocalCueMatrix();
        pcue = squeeze(obj.D.local_cue_matrix(:, q, :));
        if P == 1
            pcue = pcue(:);
        end
        obj.D.probability_cue = pcue;
        obj.D.predicted_probabilities = obj.normalizeColumns(obj.D.prior_probabilities .* pcue);
    end
end
