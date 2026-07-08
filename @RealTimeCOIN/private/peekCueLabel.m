function q = peekCueLabel(obj, raw)
%PEEKCUELABEL Integer label a raw cue would take, without mutating state.
%   q = peekCueLabel(obj, raw) returns the 1-based integer label of the raw
%   cue value raw. If raw already exists in obj.cue_values its existing label
%   is returned; if it is new, the next unused label (numel(cue_values) + 1) is
%   returned but cue_values is NOT modified. Returns [] for an empty raw. This
%   is the read-only counterpart of consumePendingCue, used by preview queries.
    if isempty(raw)
        q = [];
        return;
    end
    idx = find(arrayfun(@(x) isequal(x, raw), obj.cue_values), 1);
    if isempty(idx)
        q = numel(obj.cue_values) + 1;
    else
        q = idx;
    end
end
