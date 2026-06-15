function [W, M, V] = previewPredictiveFeedback(obj, q)
    Cmax = obj.max_contexts + 1;
    P = obj.num_particles;
    prior = zeros(Cmax, P);
    for p = 1:P
        prior(:,p) = obj.D.local_transition_matrix(obj.D.context(p), :, p)';
    end
    prior = obj.normalizeColumns(prior);

    if isempty(q)
        W = prior;
    else
        qCol = min(q, size(obj.D.local_cue_matrix, 2));
        pcue = squeeze(obj.D.local_cue_matrix(:, qCol, :));
        if P == 1
            pcue = pcue(:);
        end
        W = obj.normalizeColumns(prior .* pcue);
    end

    Mstate = obj.D.retention .* obj.D.state_filtered_mean + obj.D.drift;
    Vstate = obj.D.retention.^2 .* obj.D.state_filtered_var + obj.sigma_process_noise^2;
    for p = 1:P
        novel = min(obj.D.C(p) + 1, Cmax);
        if obj.D.C(p) < obj.max_contexts
            Mstate(novel,p) = obj.stationaryStateMean(obj.D.retention(novel,p), obj.D.drift(novel,p));
            Vstate(novel,p) = obj.stationaryStateVar(obj.D.retention(novel,p));
        end
    end
    M = Mstate + obj.D.bias;
    V = Vstate + obj.observationVariance();
end
