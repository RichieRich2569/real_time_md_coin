function [Km, modalMask, modalIdx, weights] = selectModalContexts(obj)
    P = obj.num_particles;
    cards = obj.D.C(:)';
    Km = obj.modalCardinality(cards);
    modalMask = cards == Km;
    modalIdx = find(modalMask);
    if isempty(modalIdx)
        modalIdx = 1:P;
        modalMask = true(1, P);
        Km = min(max(cards), obj.max_contexts);
    end
    weights = ones(1, numel(modalIdx)) ./ max(numel(modalIdx), 1);
end
