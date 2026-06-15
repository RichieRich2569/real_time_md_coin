function p = normalizeProbability(~, p)
    p = double(p(:)');
    p(~isfinite(p) | p < 0) = 0;
    s = sum(p);
    if s <= 0
        p = ones(size(p)) ./ max(numel(p), 1);
    else
        p = p ./ s;
    end
    p = max(p, realmin);
    p = p ./ sum(p);
end
