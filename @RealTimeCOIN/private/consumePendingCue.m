function q = consumePendingCue(obj)
%CONSUMEPENDINGCUE Resolve and clear the staged cue into an integer column label.
%   q = consumePendingCue(obj) reads the raw cue staged by observe_q into
%   obj.pending_q, clears pending_q, and maps the raw value to a 1-based
%   integer label q indexing obj.cue_values. A previously unseen raw value is
%   appended to cue_values (assigning it the next label), and the backing cue
%   columns are grown via ensureCueColumn. Returns [] when no cue was staged.
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
