function state = serializableState(obj)
%SERIALIZABLESTATE Capture the full model state as a plain struct for saving.
%   state = serializableState(obj) collects everything needed to reconstruct
%   the model into a plain struct suitable for save/load and saveModel: every
%   public property (except the dependent Trial, which is derived from trial)
%   under state.properties, plus the private particle store D, the staged cue
%   pending_q, the trial counter, the cue_values lookup, and the alignment
%   bookkeeping (state_version, alignment_seed). restoreSerializableState is
%   the inverse.
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
