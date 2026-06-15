function q = peekCueLabel(obj, raw)
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
