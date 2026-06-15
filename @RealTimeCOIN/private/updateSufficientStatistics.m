function updateSufficientStatistics(obj, y, q)
    Cmax = obj.max_contexts + 1;
    P = obj.num_particles;
    idx = sub2ind([Cmax, Cmax, P], obj.D.previous_context, obj.D.context, 1:P);
    obj.D.n_context(idx) = obj.D.n_context(idx) + 1;

    if ~isempty(q)
        obj.ensureCueColumn(q);
        idxCue = sub2ind(size(obj.D.n_cue), obj.D.context, q * ones(1, P), 1:P);
        obj.D.n_cue(idxCue) = obj.D.n_cue(idxCue) + 1;
    end

    if obj.trial > 0
        xAug = ones(Cmax, P, 2);
        xAug(:,:,1) = obj.D.previous_x_dynamics;
        observedRows = squeeze(sum(obj.D.n_context, 2)) > 0;
        ss1 = obj.D.x_dynamics .* xAug;
        obj.D.dynamics_ss_1 = obj.D.dynamics_ss_1 + ss1 .* observedRows;
        for a = 1:2
            for b = 1:2
                obj.D.dynamics_ss_2(:,:,a,b) = obj.D.dynamics_ss_2(:,:,a,b) + ...
                    xAug(:,:,a) .* xAug(:,:,b) .* observedRows;
            end
        end
    end

    if obj.infer_bias && ~isempty(y)
        obj.D.bias_ss_1(obj.D.i_observed) = obj.D.bias_ss_1(obj.D.i_observed) + (y - obj.D.x_bias);
        obj.D.bias_ss_2(obj.D.i_observed) = obj.D.bias_ss_2(obj.D.i_observed) + 1;
    end
end
