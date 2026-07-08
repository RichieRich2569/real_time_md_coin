function Km = modalCardinality(~, cards)
%MODALCARDINALITY Most frequent context count across particles (ties -> smallest).
%   Km = modalCardinality(obj, cards) returns the mode of the per-particle
%   context-cardinality vector cards, i.e. the number of contexts held by the
%   largest number of particles. When two counts are equally common the
%   smaller cardinality is returned, giving a stable, conservative point
%   estimate. The leading obj argument is ignored (private RealTimeCOIN method).
    vals = unique(cards(:)');
    counts = arrayfun(@(v) sum(cards == v), vals);
    best = counts == max(counts);
    Km = min(vals(best));
end
