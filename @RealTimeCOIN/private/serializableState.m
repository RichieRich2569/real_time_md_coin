function state = serializableState(obj)
    state = struct();
    props = properties(obj);
    for i = 1:numel(props)
        if ~strcmp(props{i}, 'Trial')
            state.properties.(props{i}) = obj.(props{i});
        end
    end
    state.D = obj.D;
    state.pending_q = obj.pending_q;
    state.trial = obj.trial;
    state.cue_values = obj.cue_values;
    state.state_version = obj.state_version;
    state.alignment_seed = obj.alignment_seed;
end
