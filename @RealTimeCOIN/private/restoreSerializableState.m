function restoreSerializableState(obj, state)
    names = fieldnames(state.properties);
    for i = 1:numel(names)
        obj.(names{i}) = state.properties.(names{i});
    end
    obj.D = state.D;
    if obj.state_dim > 1
        restoreMDBiasStatisticsCompatibility(obj);
    end
    obj.pending_q = state.pending_q;
    obj.trial = state.trial;
    obj.cue_values = state.cue_values;
    if isfield(state, 'state_version')
        obj.state_version = state.state_version;
    else
        obj.state_version = obj.trial;
    end
    if isfield(state, 'alignment_seed')
        obj.alignment_seed = state.alignment_seed;
    else
        obj.alignment_seed = [];
    end
    obj.invalidateContextAlignment();
end

function restoreMDBiasStatisticsCompatibility(obj)
    N = obj.state_dim;
    Cmax = obj.max_contexts + 1;
    P = obj.num_particles;
    if isfield(obj.D, 'bias_info_ss') && isfield(obj.D, 'bias_precision_ss')
        return;
    end

    obj.D.bias_info_ss = zeros(N, Cmax, P);
    obj.D.bias_precision_ss = zeros(N, N, Cmax, P);
    if ~isfield(obj.D, 'bias_ss_1') || ~isfield(obj.D, 'bias_ss_2')
        return;
    end

    Ri = obj.safeInverse(obj.observationNoiseCov());
    for p = 1:P
        for c = 1:Cmax
            obj.D.bias_info_ss(:, c, p) = Ri * obj.D.bias_ss_1(:, c, p);
            obj.D.bias_precision_ss(:, :, c, p) = obj.D.bias_ss_2(c, p) * Ri;
        end
    end
end
