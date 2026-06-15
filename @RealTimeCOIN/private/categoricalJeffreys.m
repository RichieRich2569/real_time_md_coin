function d = categoricalJeffreys(obj, p, q)
    n = max(numel(p), numel(q));
    p(end+1:n) = 0;
    q(end+1:n) = 0;
    p = obj.normalizeProbability(p);
    q = obj.normalizeProbability(q);
    d = sum((p - q) .* log(p ./ q));
    if ~isfinite(d)
        d = realmax;
    end
    d = max(d, 0);
end
