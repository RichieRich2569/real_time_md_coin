function restoreSerializableState(obj, state)
%RESTORESERIALIZABLESTATE Rebuild the model in place from a serializableState struct.
%   restoreSerializableState(obj, state) is the inverse of serializableState:
%   it writes every saved public property back onto obj, restores the particle
%   store D, the staged cue, trial counter and cue_values, and re-establishes
%   the alignment bookkeeping (defaulting state_version to trial and
%   alignment_seed to [] for structs saved before those fields existed). The
%   cached context alignment is invalidated so it is recomputed on next query.
%   For multi-dimensional models, legacy bias statistics are migrated forward
%   via restoreMDBiasStatisticsCompatibility.
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
%RESTOREMDBIASSTATISTICSCOMPATIBILITY Migrate legacy MD bias stats on load.
%   Back-fills the information-form bias sufficient statistics
%   (D.bias_info_ss, D.bias_precision_ss) for multi-dimensional models saved
%   before those fields existed, reconstructing them from the older
%   (bias_ss_1, bias_ss_2) representation via the observation precision.
%   Already-current states return unchanged.
%
%   DEPRECATED (flagged 2026-07-08): legacy load-compatibility shim. Retained so
%   old saved models keep loading; do NOT remove until pre-info-form saves are
%   confirmed obsolete. No new code should depend on the bias_ss_1/bias_ss_2
%   layout. TODO: drop this shim once the legacy save format is retired.
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
