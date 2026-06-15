function observe_q(obj, q)
    if isempty(q) || (isnumeric(q) && isscalar(q) && isnan(q))
        obj.pending_q = [];
        return;
    end
    obj.pending_q = q;
end
