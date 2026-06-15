function q = consumePendingCue(obj)
    raw = obj.pending_q;
    obj.pending_q = [];
    if isempty(raw)
        q = [];
        return;
    end
    idx = find(arrayfun(@(x) isequal(x, raw), obj.cue_values), 1);
    if isempty(idx)
        obj.cue_values(end+1) = raw;
        q = numel(obj.cue_values);
    else
        q = idx;
    end
    obj.ensureCueColumn(q);
end
