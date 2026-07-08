function p = normalizeProbability(~, p)
%NORMALIZEPROBABILITY Normalise a vector into a strictly positive probability row.
%   p = normalizeProbability(obj, p) returns p as a 1-by-N row that sums to 1.
%   Non-finite and negative entries are treated as zero; an all-zero (or empty)
%   input falls back to the uniform distribution. Every entry is floored at
%   realmin and the vector is renormalised so that downstream log/divide
%   operations never see an exact zero. The leading obj argument is ignored
%   (this is a private RealTimeCOIN method invoked as obj.normalizeProbability).
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
