function Km = modalCardinality(~, cards)
    vals = unique(cards(:)');
    counts = arrayfun(@(v) sum(cards == v), vals);
    best = counts == max(counts);
    Km = min(vals(best));
end
