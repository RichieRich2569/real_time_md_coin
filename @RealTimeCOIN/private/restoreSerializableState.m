function restoreSerializableState(obj, state)
    names = fieldnames(state.properties);
    for i = 1:numel(names)
        obj.(names{i}) = state.properties.(names{i});
    end
    obj.D = state.D;
    obj.pending_q = state.pending_q;
    obj.trial = state.trial;
    obj.cue_values = state.cue_values;
    if isfield(state, 'state_version')
        obj.state_version = state.state_version;
    else
        obj.state_version = obj.trial;
    end
    obj.invalidateContextAlignment();
end
